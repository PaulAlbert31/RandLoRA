import os
import torch
import torch.nn as nn
import torchvision
import timm
import src.rand_lora as rand_lora
from src.utils import convert_to_rgb

WEIGHTS={
    "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
    "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
    "base_sam":"vit_base_patch16_224.sam", # 1k
    "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
    "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
    "base_deit":"deit_base_distilled_patch16_224", # 1k
    "large":"google/vit-large-patch16-224",
    "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
    "large_dino":"facebook/dinov2-large",
    "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
    "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
    "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
    "giant_clip":"vit_giant_patch14_clip_224.laion2b",
    "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
}


def get_timm_model(args):
    import os
    os.environ["HUGGINGFACE_HUB_CACHE"] = args.openclip_cachedir
    if 'dino' in args.model:
        return torch.hub.load('facebookresearch/dinov2', args.model)
    return timm.create_model(WEIGHTS[args.model], pretrained=True)

class TimmModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_timm_model(args)

        #CLIP transforms
        self.train_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.9, 1), ratio=(0.75, 1.3333), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            convert_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.val_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.CenterCrop(size=224),
            convert_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        self.args=args

    def forward(self, x, norm=False):
        logits = self.model(x)
        if norm:
            logits = logits / logits.norm(dim=-1, keepdim=True)
        return logits
    
class TimmRandLoRA(TimmModel):
    def __init__(self, args):
        super().__init__(args)
        self.apply_peft()

    def state_dict(self, **kwargs):
        state_dict = {k:v for k, v in super().state_dict(**kwargs).items() if 'loramat' in k}
        state_dict["seed"] = self.args.seed
        state_dict["basis"] = self.basis
        return state_dict

    def apply_peft(self):
        state_dict = self.model.state_dict()
        basis = rand_lora.generate_basis(state_dict, self.args.rank, self.args.num_basis, self.args.seed, shared=self.args.param_type=="randlora", random=not self.args.sparse, dist_type=self.args.dist_type, sparcity=self.args.sparcity)
        
        self.basis = basis
        rank = self.args.rank
        for layer_index, resblock in enumerate(self.model.blocks):
            if hasattr(resblock, 'attn') and not self.args.mlp_only:
                attn = resblock.attn.qkv
                in_feats = attn.weight.shape[1]
                out_feats = attn.weight.shape[0]
                rand_lora_attn = rand_lora.Linear(in_feats, out_feats, bias=attn.bias is not None, basis=basis, param_type=self.args.param_type, seed=self.args.seed)
                resblock.attn.qkv = rand_lora_attn

            if hasattr(resblock, 'mlp') and not self.args.attn_only:
                linear = resblock.mlp
                c_fc = linear.fc1
                c_proj = linear.fc2
                in_feats = c_fc.weight.shape[1]
                out_feats = c_fc.weight.shape[0]
                rand_lora_fc = rand_lora.Linear(in_feats, out_feats, bias=c_fc.bias is not None, basis=basis, param_type=self.args.param_type, seed=self.args.seed)
                in_feats = c_proj.weight.shape[1]
                out_feats = c_proj.weight.shape[0]
                rand_lora_proj = rand_lora.Linear(in_feats, out_feats, bias=c_proj.bias is not None, basis=basis, param_type=self.args.param_type, seed=self.args.seed)
                
                resblock.mlp.fc1 = rand_lora_fc
                resblock.mlp.fc2 = rand_lora_proj

        self.model.load_state_dict(state_dict, strict=False)
        self.basis = basis
        for n, p in self.model.named_parameters():
            if 'loramat' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                
