import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates
from src.heads import build_classification_head
from src.clip_encoder import CLIP as CLIP_base
from src.clip_encoder import CLIPRandLoRA
from src.timm_encoder import TimmModel, TimmRandLoRA

def get_model(args):
    if args.full_clip:
        return CLIPVL(args)
    if args.lp_clip:
        return CLIPLP(args)
    if args.model not in ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-g-14", "ViT-H-14"]:#open_clip encoders
        return TimmClassifier(args)    
    return CLIPClassifier(args)

def get_encoder(args):
    if args.model not in ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-g-14", "ViT-H-14"]:#open_clip encoders
        if args.rand_lora:
            return TimmRandLoRA(args)
        return TimmModel(args)
    
    if args.rand_lora:
        return CLIPRandLoRA(args)    
    return CLIP_base(args)


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = get_encoder(args)
        self.encoder.model.visual.val_preprocess = self.encoder.val_preprocess
        self.encoder.model.visual.train_preprocess = self.encoder.train_preprocess
        self.image_encoder = self.encoder.model.visual
        self.text_encoder = self.encoder.model.encode_text
        self.tokenizer = open_clip.get_tokenizer(args.model)
        dataset = get_dataset(args.train_dataset, None, location=args.data_location)
        self.class_names = dataset.classnames
        self.templates = get_templates(args.train_dataset)
        self.classifier = nn.Identity()
        self.classifier.out_features = len(self.class_names)
        del dataset

    def forward(self, x, y=None, multi_prompt=False):
        feats = self.image_encoder(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        text = torch.arange(len(self.class_names))

        if y is None and multi_prompt:
            text = self.tokenizer([f'{t(self.class_names[i])}' for i in text for t in self.templates])
        else:
            text = self.tokenizer([f'An image of {self.class_names[i]}' for i in text])
        feats_text = self.text_encoder(text.to(x.device))
        feats_text = feats_text / feats_text.norm(dim=-1, keepdim=True)
        if y is None and multi_prompt:
            feats_text = torch.split(feats_text, len(self.templates))
            feats_text = torch.cat([f.mean(0, keepdim=True) for f in feats_text])            
            feats_text = feats_text / feats_text.norm(dim=-1, keepdim=True)

        sims = self.encoder.model.logit_scale * feats @ feats_text.T

        if y is None:
            return sims
        
        return F.cross_entropy(sims, y)

    def update_dataset(self, dataset_name):
        dataset = get_dataset(dataset_name, None, location=self.args.data_location)
        self.class_names = dataset.classnames
        self.templates = get_templates(dataset_name)
        self.classifier.out_features = len(self.class_names)
        

class CLIPVL(CLIP):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x, y=None, multi_prompt=False):
        feats = self.image_encoder(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        if y is None:
            text = self.tokenizer([f'An image of {self.class_names[i]}' for i in torch.arange(len(self.class_names))])
        else:
            if isinstance(y[0], str):
                text = self.tokenizer([f'{i}' for i in y])
                y = F.one_hot(torch.arange(len(y))).to(x)
            else:
                text = self.tokenizer([f'An image of {self.class_names[i]}' for i in y])
                y = F.one_hot(y).to(x)
            y = y @ y.T
            y = torch.argmax(y @ y.T, dim=-1)
            
        feats_text = self.text_encoder(text.to(x.device))
        feats_text = feats_text / feats_text.norm(dim=-1, keepdim=True)        

        sims = self.encoder.model.logit_scale * feats @ feats_text.T

        if y is None:
            return sims
        return F.cross_entropy(sims, y)
        

        
class CLIPClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = get_encoder(args)
        self.classifier = build_classification_head(self.encoder.model, args)

        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x, y=None):
        feats = self.encoder.model.visual(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = self.classifier(feats)
        if y is not None:
            return F.cross_entropy(logits, y)
        return logits

    def update_dataset(self, dataset_name):
        prev_dataset = self.args.train_dataset
        self.args.train_dataset = dataset_name
        self.classifier = build_classification_head(self.encoder.model, self.args).to(self.classifier.weight)
        for p in self.classifier.parameters():
            p.requires_grad = False

class CLIPLP(nn.Module):
    # CLIP with a learnable classifier
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = get_encoder(args)
        self.classifier = build_classification_head(self.encoder.model, args)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data *= 0.
        
    def forward(self, x, y=None):
        feats = self.encoder.model.visual(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = self.classifier(feats)
        if y is not None:
            return F.cross_entropy(logits, y)
        return logits

    def update_dataset(self, dataset_name):
        prev_dataset = self.args.train_dataset
        self.args.train_dataset = dataset_name
        self.classifier = build_classification_head(self.encoder.model, self.args).to(self.classifier.weight)
        for p in self.classifier.parameters():
            p.requires_grad = False



class TimmClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        if not isinstance(self.encoder.model.head, nn.Identity):
            self.encoder.model.head = nn.Identity()
        in_feats = self.encoder.model.norm.weight.shape[0]
        dataset = get_dataset(args.train_dataset, None, location=args.data_location)
        
        self.classifier = nn.Linear(in_feats, len(dataset.classnames), bias=None)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        
    def forward(self, x, y=None):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        if y is not None:
            return F.cross_entropy(logits, y)
        return logits

    def update_dataset(self, dataset_name):
        return

