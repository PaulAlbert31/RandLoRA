'''
Based on the loratorch project: https://github.com/Baijiong-Lin/LoRA-Torch
'''

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
import math
import torch.nn.functional as F

def gen_mats(seed, n, in_dim, out_dim, rank, device='cpu', shared=False, dist_type='uniform'):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if dist_type == 'uniform':
        basis_b = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((in_dim, rank), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        if shared:
            basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(1)], dim=0)
        else:
            basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
    elif dist_type == 'normal':
        basis_b = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((in_dim, rank), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        if shared:
            basis_a = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((rank, out_dim), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(1)], dim=0)
        else:
            basis_a = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((rank, out_dim), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        
    basis_b, basis_a = basis_b / basis_b.std(), basis_a / basis_a.std()
    return basis_b, basis_a

def gen_mats_sparse(seed, n, in_dim, out_dim, rank, device='cpu', shared=False, sparcity=6):
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

    sparcity = 1/sparcity
    
    basis_b[basis_b_ < sparcity] = -1
    basis_b[basis_b_ > 1-sparcity] = 1
    
    basis_a[basis_a_ < sparcity] = -1
    basis_a[basis_a_ > 1-sparcity] = 1

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
       

def generate_basis(state_dict, rank, n, seed, shared=False, random=False, dist_type='uniform', sparcity=6):
    basis = {}
    tbar = tqdm(state_dict.items())
    tbar.set_description("Computing othogonal basis")
    max_in_out = (0, 0)
    max_rank = 0
    all_shapes = []
    for k, v in tbar:
        if 'weight' in k:
            if 'attn' in k or 'mlp' in k:
                max_in_out = (max(v.shape[0], max_in_out[0]), max(v.shape[1], max_in_out[1]))
                max_rank = max(max_rank, min(v.shape))
    if random:
        lora_b, lora_a = gen_mats(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared, dist_type=dist_type)
    else:#Sparse
        lora_b, lora_a = gen_mats_sparse(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared, sparcity=sparcity)
    
    return nn.Parameter(lora_b, requires_grad=False), nn.Parameter(lora_a, requires_grad=False) #Declaring the bases as non-trainable parmeters enables memory sharing

def mark_only_mat_as_trainable(model):
    for n, p in model.named_parameters():
        if 'loramat' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int,
        lora_alpha: int,
        seed:int,
        n: int,
        fan_in_fan_out: bool = False,
        param_type='randlora',

    ):
        self.r = r
        self.n = n
        self.seed = seed
        self.lora_alpha = lora_alpha
        self.param_type = param_type
        
        if self.r > 0:
            #Base
            self.scaling = self.lora_alpha / self.r
            if self.param_type == 'nola':#NoLA
                self.scaling *= 1
            elif self.param_type in ['randlora', 'vera']:
                self.scaling *= 10
                
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}
        
    def _apply(self, fn):
        new_self = super()._apply(fn)
        if hasattr(self, 'basis_b'):
            new_self.basis_a, new_self.basis_b = fn(new_self.basis_a), fn(new_self.basis_b)
        return new_self   
              
    def register_param(self):
        r"""Register matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            shape = self.in_features, self.out_features
            if self.param_type == 'nola':
                self.register_parameter(lora_name, nn.Parameter(eval(f'self.{param_name}').new_zeros((2, self.n))))
                eval(f'self.{lora_name}').data = torch.ones(eval(f'self.{lora_name}').data.shape) / self.n
            elif self.param_type == 'vera':
                self.register_parameter(lora_name, nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r))))
                self.register_parameter(lora_name+'_vect', nn.Parameter(eval(f'self.{param_name}').new_zeros((shape[0]))))
                eval(f'self.{lora_name}').data = torch.ones(eval(f'self.{lora_name}').shape) / 10
            elif self.param_type == 'randlora':
                self.register_parameter(lora_name+'_B', nn.Parameter(eval(f'self.{param_name}').new_zeros((self.n, self.r))))
                self.register_parameter(lora_name+'_A', nn.Parameter(eval(f'self.{param_name}').new_zeros((self.n, min(shape)))))#Train on the smaller matrix side
                eval(f'self.{lora_name}_B').data = torch.ones(eval(f'self.{lora_name}_B').shape) / max(eval(f'self.{lora_name}_B').shape)
            elif self.param_type == 'lora':
                self.register_parameter(lora_name+'_B', nn.Parameter(eval(f'self.{param_name}').new_zeros((shape[0], self.r))))
                self.register_parameter(lora_name+'_A', nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, shape[1]))))
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_A'), a=math.sqrt(5))
            else:
                raise NotImplementedError
            #Original weight as non trainable
            eval(f'self.{param_name}').requires_grad = False
            
    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        in_dim, out_dim = self.in_features, self.out_features
        if self.param_type == 'lora':
            lora_b, lora_a = eval(f'self.{lora_name}_B'), eval(f'self.{lora_name}_A')
            if not self.merged:
                return lora_b, lora_a
            lora_b, lora_a = lora_b.permute(1,0,2).flatten(start_dim=1), lora_a.permute(1,0,2).flatten(end_dim=1)
            lora = (lora_b @ lora_a)
        elif self.param_type == 'nola':
            coefficients = eval(f'self.{lora_name}').to(eval('self.basis_b'))
            basis_b = torch.einsum('blk,b->lk', eval('self.basis_b')[:, :in_dim, :], coefficients[0])
            basis_a = torch.einsum('blk,b->lk', eval('self.basis_a')[:, :, :out_dim], coefficients[1])
            basis_b, basis_a = basis_b.unsqueeze(0), basis_a.unsqueeze(0)
            if not self.merged:
                return basis_b, basis_a            
            lora = basis_b.sum(0) @ basis_a.sum(0) 
        elif self.param_type == 'vera':
            coefficients = eval(f'self.{lora_name}').to(eval('self.basis_b'))
            basis_b = coefficients[None, None, :] * eval('self.basis_b')[:, :in_dim, :] * eval(f'self.{lora_name}_vect')[None, :, None]
            basis_a = eval('self.basis_a')[:, :, :out_dim]
            if not self.merged:
                return basis_b, basis_a
            lora = basis_b @ basis_a
            lora = lora.sum(0)
        elif self.param_type == 'randlora':
            b, a = eval(f'self.{lora_name}_B'), eval(f'self.{lora_name}_A')
            min_dim, max_dim = max(in_dim, out_dim), min(in_dim, out_dim)            
            lora_b, lora_a = eval('self.basis_b')[:self.n, :min_dim, :], LoadParams.apply(eval('self.basis_a')[:self.n, :, :max_dim], b, a)                
            if not self.merged:
                return lora_b, lora_a            
            lora_b, lora_a = lora_b.permute(1,0,2).flatten(start_dim=1), lora_a.permute(1,0,2).flatten(end_dim=1)
            lora = lora_b.to(lora_a) @ lora_a
        else:
            raise NotImplementedError
        
        return lora

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            delta = self.merge_BA(param_name) * self.scaling
            if delta.shape == eval(f'self.{param_name}').data.shape:
                eval(f'self.{param_name}').data += delta
            else:
                eval(f'self.{param_name}').data += delta.T
                
    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            delta = self.merge_BA(param_name) * self.scaling
            if delta.shape == eval(f'self.{param_name}').data.shape:
                eval(f'self.{param_name}').data -= delta
            else:
                eval(f'self.{param_name}').data -= delta.T
            
    def lora_train(self, mode: bool = True):
        if mode:
            # Make sure that the weights are merged
            if self.merged and self.r > 0:
                self.sub_lora_data()
                self.merged = False
        else:
            # Merge the weights and mark it
            if not self.merged and self.r > 0:
                self.merged = True
                self.add_lora_data()
                                   
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            basis: Tuple[torch.tensor, torch.tensor],
            param_type: str,
            seed: int,
            lora_alpha: int = 1, 
            fan_in_fan_out: bool = False,
            bias=True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        self.basis_b, self.basis_a = basis
        n, _, r = self.basis_b.shape

        if param_type == "randlora":
            #Number of bases = rank / r
            n = int(min(self.basis_b.shape[0], self.in_features//r+1, self.out_features//r+1))

        self.in_features = in_features
        self.out_features = out_features
            
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out, param_type=param_type, seed=seed, n=n)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w_loramat'}
        self.register_param()
        nn.Linear.reset_parameters(self)        
        self.weight.data = self.transpose(self.weight.data)
        
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):        
        if self.r > 0 and not self.merged:
            lora_b, lora_a = self.merge_BA('weight')
            result = nn.Linear.forward(self, x, **kwargs)
            shape = x.shape
            lora_b, lora_a = lora_b.permute(1,0,2).flatten(start_dim=1), lora_a.permute(1,0,2).flatten(end_dim=1)               
                
            if x.shape[-1] != lora_b.shape[0]:
                lora_b, lora_a = lora_a.mT, lora_b.mT
                
            lora = (x @ lora_b) @ lora_a                    
                
            return result + self.scaling * lora

        return nn.Linear.forward(self, x, **kwargs)


class MultiheadAttention(nn.MultiheadAttention, LoRALayer):
    # LoRA implemented in a MultiheadAttention layer
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        basis: Tuple[torch.tensor, torch.tensor],
        param_type: str,
        seed: int,
        enable_lora: list = ['q', 'k', 'v', 'o'],
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.MultiheadAttention.__init__(self, embed_dim, num_heads, **kwargs)
        self.basis_b, self.basis_a = basis        
        n, _, r = self.basis_b.shape

        self.in_features = embed_dim * 3
        self.out_features = embed_dim

        if param_type == "randlora":
            #Number of bases = rank / r
            n = int(min(self.basis_b.shape[0], self.out_features//r+1))
            
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, param_type=param_type, seed=seed, n=n)

        # Actual trainable parameters
        if self.r > 0:
            if 'o' in enable_lora:
                self.params_with_lora.update({'out_proj.weight': 'out_proj_w_loramat'})
            if 'q' in enable_lora:
                self.params_with_lora.update({'in_proj_weight': 'in_proj_w_loramat'})
            nn.MultiheadAttention._reset_parameters(self)
            
        self.register_param()
                
    def train(self, mode: bool = True):
        nn.MultiheadAttention.train(self, mode)
        self.lora_train(mode)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
        if self.r > 0 and not self.merged:
            #We can't optimize matmul the same way it is done with Linear layers because nn.MultiheadAttention contains 2 Linear layers
            lora_b, lora_a = self.merge_BA('in_proj_weight')
            lora = lora_b.permute(1,0,2).flatten(start_dim=1) @ lora_a.permute(1,0,2).flatten(end_dim=1) * self.scaling
            p = set_param(self, 'in_proj_weight', mode='get')
            # detach() is very important here
            if lora.shape == p.shape:
                p_new = p.detach() + lora
            else:
                p_new = p.detach() + lora.T
                
            set_param(self, 'in_proj_weight', param=p_new, mode='update')
            result = nn.MultiheadAttention.forward(self, query, key, value, **kwargs)
            
            if lora.shape == p.shape:
                self.in_proj_weight.data -= lora
            else:
                self.in_proj_weight.data -= lora.T
            return result
        
        return nn.MultiheadAttention.forward(self, query, key, value, **kwargs)

