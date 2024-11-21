import os
import torch
import torch.nn as nn
from tqdm import tqdm
import open_clip
from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset

def build_classification_head(model, args, device='cuda'):
    dataset_name = args.train_dataset
    filename = os.path.join(args.save, f"head_{dataset_name}_{args.pretrained}.pt")
    if os.path.isfile(filename):
        zeroshot_weights = torch.load(filename)
        classification_head = ClassificationHead(weights=zeroshot_weights)
        return classification_head
        
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(dataset_name, None, location=args.data_location)
    model.eval().to(device)
    
    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames, disable=args.no_tqdm):
            texts = []
            for t in template:
                texts.append(t(classname))
                
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
        
    os.makedirs('/'.join(filename.split('/'))[:-1], exist_ok=True)
    torch.save(zeroshot_weights, filename)
    classification_head = ClassificationHead(weights=zeroshot_weights)
    return classification_head


class ClassificationHead(nn.Linear):
    def __init__(self, weights):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        if weights is not None:
            self.weight = nn.Parameter(weights.clone())
        self.bias = nn.Parameter(torch.zeros_like(self.bias))
            
    def forward(self, inputs):
        return super().forward(inputs)
