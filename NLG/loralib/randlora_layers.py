#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


def gen_mats(seed, n, in_dim, out_dim, rank, device='cpu', shared=False):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    basis_b = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((in_dim, rank), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
    if shared:
        basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(1)], dim=0)
    else:
        basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        
    basis_b, basis_a = basis_b / basis_b.std(), basis_a / basis_a.std()
    return basis_b, basis_a

def gen_mats_sparse(seed, n, in_dim, out_dim, rank, device='cpu', shared=False):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    basis_b_ = torch.rand((n, in_dim, rank), device=device, requires_grad=False, dtype=torch.float32)
    basis_b = torch.zeros((n, in_dim, rank), device=device, requires_grad=False, dtype=torch.float32)
    if shared:
        basis_a_ = torch.rand((1, rank, out_dim), device=device, requires_grad=False, dtype=torch.float32)
        basis_a = torch.zeros((1, rank, out_dim), device=device, requires_grad=False, dtype=torch.float32)
    else:
        basis_a_ = torch.rand((n, rank, out_dim), device=device, requires_grad=False, dtype=torch.float32)
        basis_a = torch.zeros((n, rank, out_dim), device=device, requires_grad=False, dtype=torch.float32)
    
    basis_b[basis_b_ < 1/6] = -1
    basis_b[basis_b_ > 5/6] = 1
    
    basis_a[basis_a_ < 1/6] = -1
    basis_a[basis_a_ > 5/6] = 1

    basis_b, basis_a = basis_b / basis_b.std(), basis_a / basis_a.std()
    return basis_b, basis_a

class LoadParams(torch.autograd.Function):
    #Memory efficent update with the shared A base
    @staticmethod    
    def forward(ctx, basis_a, b ,a):
        Out = b[:, :, None] * basis_a * a[:, None, :]
        ctx.save_for_backward(basis_a, b, a)
        return Out
    
    @staticmethod
    def backward(ctx, grad_output):
        basis_a, b, a = ctx.saved_tensors
        basis_a, b, a = basis_a.to(grad_output.dtype), b.to(grad_output.dtype), a.to(grad_output.dtype)
        grad_a = torch.einsum('bkj,bkj,bk->bj', grad_output, basis_a, b)
        grad_b = torch.einsum('bkj,bkj,bj->bk', grad_output, basis_a, a)
        return None, grad_b, grad_a

def generate_basis(rank, seed, n=None, max_in_out=None, state_dict=None, shared=True, sparse=False):
    basis = {}
    if max_in_out is None:
        max_in_out = (0, 0)
        max_rank = 0
        all_shapes = []
        for k, v in state_dict.items():
            if 'weight' in k:
                if 'attn' in k or 'mlp' in k: #Update
                    max_in_out = (max(v.shape[0], max_in_out[0]), max(v.shape[1], max_in_out[1]))
                    max_rank = max(max_rank, min(v.shape))

    if n is None:
        n = min(max_in_out) // rank + 1        
    if not sparse:
        lora_b, lora_a = gen_mats(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared)
    else:#Sparse
        lora_b, lora_a = gen_mats_sparse(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared)
    
    return nn.Parameter(lora_b, requires_grad=False), nn.Parameter(lora_a, requires_grad=False) #Declaring the bases as non-trainable parmeters enables memory sharing

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class RandLoRAMergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        basis: nn.Parameter = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.basis_b, self.basis_a = basis
        self.n, _, r = self.basis_b.shape
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.randlora_lambda = nn.Parameter(self.weight.new_zeros((self.n, self.r * sum(enable_lora))))
            self.randlora_gamma  = nn.Parameter(self.weight.new_zeros((self.n, (out_features // len(enable_lora) * sum(enable_lora)))))
            self.scaling = self.lora_alpha / self.r / math.sqrt(self.n)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'randlora_gamma'):
            self.randlora_lambda.data = torch.zeros(self.randlora_lambda.shape)
            self.randlora_gamma.data = torch.ones(self.randlora_gamma.shape) / max(self.randlora_gamma.shape)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_basis(self):
        b, a = self.randlora_lambda, self.randlora_gamma
        max_dim, min_dim = self.in_features, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        lora_a = None
        for i in range(sum(self.enable_lora)):
            dim_b, dim_a = b.shape[1] // sum(self.enable_lora), a.shape[1] // sum(self.enable_lora)
            param_update = LoadParams.apply(self.basis_a[:self.n, :, :max_dim], b[:, dim_b*i:dim_b*(i+1)], a[:, dim_a*i:dim_a*(i+1)])
            if lora_a is None:
                lora_a = param_update
            else:
                lora_a = torch.cat((lora_a, param_update), dim=1)
                
        lora_b = self.basis_b[:self.n, :min_dim, :]
        lora_b, lora_a = lora_b.permute(1,0,2).flatten(start_dim=1).to(lora_a), lora_a.permute(1,0,2).flatten(end_dim=1)
        
        return lora_b, lora_a            
        
    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        lora_b, lora_a = self.merge_basis()
        delta_w = F.conv1d(
            lora_a.unsqueeze(0), 
            lora_b.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling                
            return result

class ConvRandLoRA(nn.Module, LoRALayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, basis = None, **kwargs):
        super(ConvRandLoRA, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.nf = in_features
        self.in_features, self.out_features = out_features, in_features
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.basis_b, self.basis_a = basis
        self.n, _, r = self.basis_b.shape
        # Actual trainable parameters
        if r > 0:
            self.randlora_lambda = nn.Parameter(self.weight.new_zeros((self.n, self.r )))
            self.randlora_gamma  = nn.Parameter(self.weight.new_zeros((self.n, min(out_features, in_features))))
            self.scaling = self.lora_alpha / self.r / math.sqrt(self.n)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        self.merged = False

    def merge_basis(self):
        b, a = self.randlora_lambda, self.randlora_gamma
        max_dim, min_dim = max(self.in_features, self.out_features), min(self.in_features, self.out_features)        
        lora_a = LoadParams.apply(self.basis_a[:self.n, :, :min_dim], b, a)
        
        lora_b = self.basis_b[:self.n, :max_dim, :]
        lora_b, lora_a = lora_b.permute(1,0,2).flatten(start_dim=1).to(lora_a), lora_a.permute(1,0,2).flatten(end_dim=1)
        
        return lora_b, lora_a

    def reset_parameters(self):
        if hasattr(self, 'randlora_gamma'):
            self.randlora_lambda.data = torch.zeros(self.randlora_lambda.shape)
            self.randlora_gamma.data = torch.ones(self.randlora_gamma.shape) / max(self.randlora_gamma.shape)

    def train(self, mode=True):
        super(ConvRandLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    lora_B, lora_A = self.merge_basis()
                    # Make sure that the weights are not merged
                    self.weight.data -= (lora_B @ lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    lora_B, lora_A = self.merge_basis()
                    # Merge the weights and mark it
                    self.weight.data += (lora_B @ lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            size_out = x.size()[:-1] + (self.nf,)
            lora_B, lora_A = self.merge_basis()
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight + (lora_B @ lora_A).view(self.weight.shape) * self.scaling)
            x = x.view(*size_out)            
        else:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
        return x
