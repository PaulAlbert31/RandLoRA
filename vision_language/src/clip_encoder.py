import os
import torch
import torch.nn as nn
import open_clip
import src.rand_lora as rand_lora
import math

class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, cache_dir=args.openclip_cachedir
        )
        self.args = args
        
    def forward(self, x, norm=True):
        logits = self.model.encode_image(x)
        if norm:
            logits = logits / logits.norm(dim=-1, keepdim=True) 
        return logits
              
class CLIPRandLoRA(CLIP):
    def __init__(self, args):
        super().__init__(args)
        self.apply_peft(which=['self.model.visual.transformer.resblocks', 'self.model.transformer.resblocks'])
        
    def state_dict(self, **kwargs):
        state_dict = {k:v for k, v in super().state_dict(**kwargs).items() if 'loramat' in k}
        state_dict["seed"] = self.args.seed
        state_dict["basis"] = self.basis
        return state_dict

    def apply_peft(self, which=['self.model.visual.transformer.resblocks', 'self.model.transformer.resblocks']):
        state_dict = self.model.state_dict()
        basis = rand_lora.generate_basis(state_dict, self.args.rank, self.args.num_basis, self.args.seed, shared=self.args.param_type=="randlora", random=not self.args.sparse, dist_type=self.args.dist_type, sparcity=self.args.sparcity)
            
        self.basis = basis
            
        rank = self.args.rank
        for w in which:
            for layer_index, resblock in enumerate(eval(w)):
                if hasattr(resblock, 'attn') and not self.args.mlp_only:
                    multihead = resblock.attn
                    embed_dim = multihead.embed_dim
                    num_heads = multihead.num_heads
                    enable = ['q', 'k', 'v']
                    rand_lora_multihead = rand_lora.MultiheadAttention(embed_dim, num_heads, basis=basis, param_type=self.args.param_type, seed=self.args.seed, enable_lora=enable)                        
                    resblock.attn = rand_lora_multihead

                if hasattr(resblock, 'mlp') and not self.args.attn_only:
                    linear = resblock.mlp
                    c_fc = linear.c_fc
                    c_proj = linear.c_proj
                    in_feats = c_fc.weight.shape[1]
                    out_feats = c_fc.weight.shape[0]
                    rand_lora_fc = rand_lora.Linear(in_feats, out_feats, basis=basis, param_type=self.args.param_type, seed=self.args.seed)
                    in_feats = c_proj.weight.shape[1]
                    out_feats = c_proj.weight.shape[0]
                    rand_lora_proj = rand_lora.Linear(in_feats, out_feats, basis=basis, param_type=self.args.param_type, seed=self.args.seed)

                    resblock.mlp.c_fc = rand_lora_fc
                    resblock.mlp.c_proj = rand_lora_proj

        self.model.load_state_dict(state_dict, strict=False)
        for n, p in self.model.named_parameters():
            if 'loramat' not in n:
                p.requires_grad = False
