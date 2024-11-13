import torch
import torch.nn as nn
import random
from PIL import ImageFilter
import math
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
#https://github.com/jiaweizzhao/GaLore/blob/master/peft_pretraining/training_utils.py

def cosine_lr(optimizer, num_training_steps, min_lr_ratio=0.01, last_epoch=-1):
    num_warmup_steps = 0#num_training_steps // 10
    cycle_length = num_training_steps

    lr_lambda = partial(
        get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def extract_names(f):
    with open(f, 'r') as fi:
        lines = fi.readlines()
        return [l.split(' ')[0] for l in lines]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class InfiniteRandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indexes, total_samples):
        super().__init__()
        self.indexes = indexes
        self.total_samples = total_samples

    def __iter__(self):
        for i in torch.randint(len(self.indexes), (self.total_samples, )):
            yield self.indexes[i]
        

    def __len__(self):
        return self.total_samples
        

    def __len__(self):
        return self.total_samples
    
def convert_to_rgb(im):
    return im.convert("RGB")
    

def rand_lora_to_lora(state_dict, basis, keys=['mlp.c_proj.bias', 'mlp.c_fc.bias', 'attn.in_proj_bias']):
    new_state_dict = {}
    _keys = list(state_dict.keys())
    for k in _keys:
        if any([ke in k for ke in keys]):
            name = k.split('bias')[0]

            out_dim = max(state_dict[name+'w_loramat_A'].shape)
            
            basis_b, basis_a = basis[f'{out_dim}']
            rank = basis_b.shape[-1]
            
            rand_lora_b, rand_lora_a = state_dict[name+'w_loramat_B'], state_dict[name+'w_loramat_A']

            in_dim = out_dim
            if 'c_proj' in name:
                in_dim *= 4
            elif 'c_fc' in name:
                in_dim *= 4
            elif 'attn' in name:
                in_dim *= 3
                

            lora_merged = (rand_lora_b[:, None, :] * basis_b[:, :in_dim, :]) @ (rand_lora_a[:, None, :] * basis_a)
            lora_merged = lora_merged.mean(0) * 10 * 4 / rank
            
            if 'c_proj' in name:
                lora_merged = lora_merged.T
            
            U, S, V = torch.linalg.svd(lora_merged)
            lora_b = U[:, :rank] 
            lora_a = torch.diag(S[:rank]) @ V[:rank, :]

            name = '.'.join(k.split('.')[:-1])
            if 'attn' in k:
                new_state_dict[name+'.qkv_lora_B'] = lora_b
                new_state_dict[name+'.qkv_lora_A'] = lora_a
            else:
                new_state_dict[name+'.w_lora_B'] = lora_b
                new_state_dict[name+'.w_lora_A'] = lora_a

    return new_state_dict


def reconstruction(w0, basis_b, basis_a, param_type):
    n, r, out_dim = basis_a.shape
    n, in_dim, r = basis_b.shape
    rand_lora_b = torch.ones((n, r), device='cuda', requires_grad=False, dtype=torch.float32) / max([n,r])
    if param_type == 'vecteff':
        rand_lora_a = torch.ones((1, out_dim), device='cuda', requires_grad=False, dtype=torch.float32) / out_dim
    elif param_type == 'vect':
        rand_lora_a = torch.ones((n, out_dim), device='cuda', requires_grad=False, dtype=torch.float32) / out_dim
        
    rand_lora_b.requires_grad = True
    rand_lora_a.requires_grad = True
    optim = torch.optim.AdamW([rand_lora_a, rand_lora_b], lr=0.001)
    
    basis_b, basis_a, rand_lora_a, rand_lora_b = basis_b.cuda(), basis_a.cuda(), rand_lora_a.cuda(), rand_lora_b.cuda()
    for i in range(100):
        w = (rand_lora_b[:, None, :] * basis_b[:, :max(w0.shape), :])
        w = w @ (rand_lora_a[:, None, :] * basis_a)
        w = w.mean(0) * 10 * r / 4
        loss = torch.norm(w - w0)
        #loss = torch.nn.functional.mse_loss(w, w0).sqrt()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return rand_lora_b.detach().cpu(), rand_lora_a.detach().cpu()

def lora_to_rand_lora(state_dict, basis, param_type, keys=['mlp.c_proj.weight', 'mlp.c_fc.weight', 'attn.in_proj_weight']):
    new_state_dict = {k:v for k,v in state_dict.items() if 'lora' not in k}
    for k in new_state_dict:
        if any([ke in k for ke in keys]):
            w = new_state_dict[k]
            basis_b, basis_a = basis[f'{min(w.shape)}']
            name = '.'.join(k.split('.')[:-1])

            if 'attn' in k:
                lora_b, lora_a = state_dict[name+'.qkv_lora_B'], state_dict[name+'.qkv_lora_A']
            else:
                lora_b, lora_a = state_dict[name+'.w_lora_B'], state_dict[name+'.w_lora_A']

            lora_merged = lora_b @ lora_a
            if lora_merged.shape[0] != basis_b.shape[1]:
                lora_merged = lora_merged.T

            rand_lora_b, rand_lora_a = reconstruction(lora_merged, basis_b, basis_a, param_type)
            
            if 'attn' in k:
                new_state_dict[name+'.in_proj_w_loramat_B'] = lora_b
                new_state_dict[name+'.in_proj_w_loramat_A'] = lora_a
            else:
                new_state_dict[name+'.w_loramat_B'] = lora_b
                new_state_dict[name+'.w_loramat_A'] = lora_a

    return new_state_dict
