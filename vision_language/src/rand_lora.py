'''
Based on the loratorch project: https://github.com/Baijiong-Lin/LoRA-Torch. Documentation generated with the help of google's aistudio
'''

import torch
import torch.nn as nn
from typing import Tuple
import math
import torch.nn.functional as F

def gen_mats(seed, n, in_dim, out_dim, rank, device='cpu', shared=False, dist_type='uniform'):
    """
    Generates a pair of basis matrices (B and A) for RandLoRA.

    Args:
        seed (int): Random seed for initialization.
        n (int): Number of basis pairs to generate.
        in_dim (int): Input dimension of the original weight matrix.
        out_dim (int): Output dimension of the original weight matrix.
        rank (int): Rank of the low-rank approximation.
        device (str, optional): Device to generate matrices on. Defaults to 'cpu'.
        shared (bool, optional): Whether to share the A matrix across all n basis pairs. Defaults to False.
        dist_type (str, optional): Distribution type for initializing matrices ('uniform' or 'normal'). Defaults to 'uniform'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the basis matrices (basis_b, basis_a).
            - basis_b: Tensor of shape (n, in_dim, rank).
            - basis_a: Tensor of shape (n or 1, rank, out_dim) depending on the 'shared' parameter.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if dist_type == 'uniform':
        # Initialize basis_b with Kaiming uniform initialization
        basis_b = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((in_dim, rank), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        if shared:
            # Initialize a single basis_a if shared is True
            basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(1)], dim=0)
        else:
            # Initialize multiple basis_a if shared is False
            basis_a = torch.cat([torch.nn.init.kaiming_uniform_(torch.empty((rank, out_dim), device=device), a=math.sqrt(5), generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
    elif dist_type == 'normal':
        # Initialize basis_b with Kaiming normal initialization
        basis_b = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((in_dim, rank), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(n)], dim=0)
        if shared:
            # Initialize a single basis_a if shared is True
            basis_a = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((rank, out_dim), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(1)], dim=0)
        else:
            # Initialize multiple basis_a if shared is False
            basis_a = torch.cat([torch.nn.init.kaiming_normal_(torch.empty((rank, out_dim), device=device), a=0, generator=gen).unsqueeze(0) for _ in range(n)], dim=0)

    # Normalize the basis matrices
    basis_b, basis_a = basis_b / basis_b.std(), basis_a / basis_a.std()
    return basis_b, basis_a

def gen_mats_sparse(seed, n, in_dim, out_dim, rank, device='cpu', shared=False, sparcity=6):
    """
    Generates sparse basis matrices (B and A) for RandLoRA.

    Args:
        seed (int): Random seed for initialization.
        n (int): Number of basis pairs to generate.
        in_dim (int): Input dimension of the original weight matrix.
        out_dim (int): Output dimension of the original weight matrix.
        rank (int): Rank of the low-rank approximation.
        device (str, optional): Device to generate matrices on. Defaults to 'cpu'.
        shared (bool, optional): Whether to share the A matrix across all n basis pairs. Defaults to False.
        sparcity (int, optional): Controls the sparsity of the matrices. Higher value means more sparsity. Defaults to 6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sparse basis matrices (basis_b, basis_a).
            - basis_b: Tensor of shape (n, in_dim, rank) with sparse values (-1, 0, 1).
            - basis_a: Tensor of shape (n or 1, rank, out_dim) depending on the 'shared' parameter, with sparse values (-1, 0, 1).
    """
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

    # Apply sparsity mask: set values close to 0 to -1, values close to 1 to 1, and others remain 0
    basis_b[basis_b_ < sparcity] = -1
    basis_b[basis_b_ > 1-sparcity] = 1

    basis_a[basis_a_ < sparcity] = -1
    basis_a[basis_a_ > 1-sparcity] = 1

    # Normalize the basis matrices
    basis_b, basis_a = basis_b / basis_b.std(), basis_a / basis_a.std()
    return basis_b, basis_a

class LoadParams(torch.autograd.Function):
    """
    A custom autograd function for performing the RandLoRA update in the forward pass
    and calculating gradients efficiently in the backward pass, especially with shared A bases.
    """
    #Memory efficent update with the shared A base
    @staticmethod
    def forward(ctx, basis_a, b ,a):
        """
        Forward pass of the LoadParams function.

        Args:
            ctx: Context object for storing tensors needed in the backward pass.
            basis_a (torch.Tensor): The A basis matrix.
            b (torch.Tensor): The trainable parameter B.
            a (torch.Tensor): The trainable parameter A.

        Returns:
            torch.Tensor: The output of the RandLoRA update: b[:, :, None] * basis_a * a[:, None, :].
        """
        Out = b[:, :, None] * basis_a * a[:, None, :]
        ctx.save_for_backward(basis_a, b, a)
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the LoadParams function.

        Args:
            ctx: Context object containing saved tensors from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[None, torch.Tensor, torch.Tensor]: Gradients with respect to the inputs (basis_a, b, a).
                The gradient for basis_a is None as it's considered a fixed basis.
        """
        basis_a, b, a = ctx.saved_tensors
        basis_a, b, a = basis_a.to(grad_output.dtype), b.to(grad_output.dtype), a.to(grad_output.dtype)
        grad_a = torch.einsum('bkj,bkj,bk->bj', grad_output, basis_a, b)
        grad_b = torch.einsum('bkj,bkj,bj->bk', grad_output, basis_a, a)
        return None, grad_b, grad_a

def generate_basis(state_dict, rank, n, seed, shared=False, random=False, dist_type='uniform', sparcity=6):
    """
    Generates the basis matrices (B and A) based on the provided model's state dictionary.

    Args:
        state_dict (dict): The state dictionary of the model.
        rank (int): Rank of the low-rank approximation.
        n (int): Number of basis pairs to generate.
        seed (int): Random seed for initialization.
        shared (bool, optional): Whether to share the A matrix across all n basis pairs. Defaults to False.
        random (bool, optional): If True, generate dense basis matrices. If False, generate sparse basis matrices. Defaults to False.
        dist_type (str, optional): Distribution type for initializing dense matrices ('uniform' or 'normal'). Defaults to 'uniform'.
        sparcity (int, optional): Controls the sparsity of the matrices if random is False. Defaults to 6.

    Returns:
        Tuple[nn.Parameter, nn.Parameter]: A tuple containing the basis matrices as non-trainable parameters.
            - lora_b: Parameter of shape (n, in_dim, rank).
            - lora_a: Parameter of shape (n or 1, rank, out_dim) depending on the 'shared' parameter.
    """
    basis = {}
    max_in_out = (0, 0)
    max_rank = 0
    all_shapes = []
    # Find the maximum input and output dimensions and maximum rank from the weight matrices in the state dictionary
    for k, v in state_dict.items():
        if 'weight' in k:
            if 'attn' in k or 'mlp' in k:
                max_in_out = (max(v.shape[0], max_in_out[0]), max(v.shape[1], max_in_out[1]))
                max_rank = max(max_rank, min(v.shape))
    if random:
        # Generate dense basis matrices
        lora_b, lora_a = gen_mats(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared, dist_type=dist_type)
    else:#Sparse
        # Generate sparse basis matrices
        lora_b, lora_a = gen_mats_sparse(seed, n, max_in_out[0], max_in_out[1], rank, shared=shared, sparcity=sparcity)

    return nn.Parameter(lora_b, requires_grad=False), nn.Parameter(lora_a, requires_grad=False) #Declaring the bases as non-trainable parmeters enables memory sharing

def mark_only_mat_as_trainable(model):
    """
    Marks only the RandLoRA matrices ('loramat' in the parameter name) as trainable, freezing other parameters.

    Args:
        model (nn.Module): The model to modify.
    """
    for n, p in model.named_parameters():
        if 'loramat' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def set_param(curr_mod, name, param=None, mode='update'):
    """
    A utility function to get or update a parameter within a module, even if it's nested.
    Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py

    Args:
        curr_mod (nn.Module): The current module.
        name (str): The name of the parameter (can be a nested name like 'layer.sublayer.weight').
        param (torch.Tensor, optional): The new parameter value if mode is 'update'. Defaults to None.
        mode (str, optional): 'update' to set a parameter, 'get' to retrieve a parameter. Defaults to 'update'.

    Returns:
        torch.Tensor or None: If mode is 'get', returns the parameter. Otherwise, returns None.
    """
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
    """
    A base class for implementing LoRA (Low-Rank Adaptation) on different types of layers.
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        seed:int,
        n: int,
        fan_in_fan_out: bool = False,
        param_type='randlora',

    ):
        """
        Initializes the LoRALayer.

        Args:
            r (int): Rank of the low-rank approximation.
            lora_alpha (int): Scaling factor for the LoRA contribution.
            seed (int): Random seed for initialization.
            n (int): Number of basis pairs.
            fan_in_fan_out (bool, optional): Set to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
            param_type (str, optional): Type of parametrization ('randlora', 'vera', 'nola', 'lora'). Defaults to 'randlora'.
        """
        self.r = r
        self.n = n
        self.seed = seed
        self.lora_alpha = lora_alpha
        self.param_type = param_type

        if self.r > 0:
            # Base scaling factor
            self.scaling = self.lora_alpha / self.r        

        # Mark the weight as unmerged initially
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def _apply(self, fn):
        """
        Applies a function to the basis matrices. Overrides the base class method to handle basis tensors.
        """
        new_self = super()._apply(fn)
        if hasattr(self, 'basis_b'):
            new_self.basis_a, new_self.basis_b = fn(new_self.basis_a), fn(new_self.basis_b)
        return new_self

    def register_param(self):
        """
        Registers the LoRA parameters (and potentially additional parameters depending on the `param_type`).
        """
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
                self.register_parameter(lora_name+'_B', nn.Parameter(eval(f'self.{param_name}').new_zeros((1, shape[0], self.r))))
                self.register_parameter(lora_name+'_A', nn.Parameter(eval(f'self.{param_name}').new_zeros((1, self.r, shape[1]))))
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_A'), a=math.sqrt(5))
            else:
                raise NotImplementedError
            #Original weight as non trainable
            eval(f'self.{param_name}').requires_grad = False
        
    def transpose(self, w: torch.Tensor):
        """
        Transposes the weight matrix if fan_in_fan_out is True.
        """
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        """
        Calculates the low-rank adaptation matrix (B @ A) based on the `param_type`.

        Args:
            param_name (str): The name of the original parameter.

        Returns:
            torch.Tensor: The low-rank adaptation matrix.
        """
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
        """
        Merges the LoRA weights into the original weight matrix, making the combined weight available for inference.
        The LoRA parameters remain differentiable for continued training.
        """
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        """
        Merges the LoRA weights into the original weight matrix in-place (not differentiable).
        This is useful for inference or saving the merged weights.
        """
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            delta = self.merge_BA(param_name) * self.scaling
            if delta.shape == eval(f'self.{param_name}').data.shape:
                eval(f'self.{param_name}').data += delta
            else:
                eval(f'self.{param_name}').data += delta.T

    def sub_lora_data(self):
        """
        Subtracts the LoRA weights from the original weight matrix in-place (not differentiable).
        This is used when unmerging weights to prepare for LoRA training.
        """
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            delta = self.merge_BA(param_name) * self.scaling
            if delta.shape == eval(f'self.{param_name}').data.shape:
                eval(f'self.{param_name}').data -= delta
            else:
                eval(f'self.{param_name}').data -= delta.T

    def lora_train(self, mode: bool = True):
        """
        Sets the training mode for the LoRA layer.

        Args:
            mode (bool, optional): If True, prepares the layer for LoRA training (unmerges weights if needed).
                                  If False, merges the weights for inference. Defaults to True.
        """
        if mode:
            # Make sure that the weights are unmerged
            if self.merged and self.r > 0:
                self.sub_lora_data()
                self.merged = False
        else:
            # Merge the weights and mark it
            if not self.merged and self.r > 0:
                self.merged = True
                self.add_lora_data()

class Linear(nn.Linear, LoRALayer):
    """
    Implements LoRA (Low-Rank Adaptation) for a linear layer.
    """
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
        """
        Initializes the LoRA Linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            basis (Tuple[torch.Tensor, torch.Tensor]): The pre-generated basis matrices (basis_b, basis_a).
            param_type (str): Type of parametrization ('randlora', 'vera', 'nola', 'lora').
            seed (int): Random seed for initialization.
            lora_alpha (int, optional): Scaling factor for the LoRA contribution. Defaults to 1.
            fan_in_fan_out (bool, optional): Set to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
            bias (bool, optional): If True, adds a bias to the linear transformation. Defaults to True.
            **kwargs: Additional keyword arguments passed to the nn.Linear constructor.
        """
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
        """
        Sets the training mode for the layer and its LoRA components.

        Args:
            mode (bool, optional): If True, sets the module in training mode. Defaults to True.
        """
        nn.Linear.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass of the LoRA Linear layer.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments passed to the nn.Linear forward method.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation with LoRA.
        """
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
    """
    Implements LoRA (Low-Rank Adaptation) for a multi-head attention layer.
    """
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
        """
        Initializes the LoRA MultiheadAttention layer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            basis (Tuple[torch.Tensor, torch.Tensor]): The pre-generated basis matrices (basis_b, basis_a).
            param_type (str): Type of parametrization ('randlora', 'vera', 'nola', 'lora').
            seed (int): Random seed for initialization.
            enable_lora (list, optional): List of attention weights to apply LoRA to (e.g., ['q', 'k', 'v', 'o']). Defaults to ['q', 'k', 'v', 'o'].
            lora_alpha (int, optional): Scaling factor for the LoRA contribution. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the nn.MultiheadAttention constructor.
        """
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
        """
        Sets the training mode for the layer and its LoRA components.

        Args:
            mode (bool, optional): If True, sets the module in training mode. Defaults to True.
        """
        nn.MultiheadAttention.train(self, mode)
        self.lora_train(mode)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
        """
        Forward pass of the LoRA MultiheadAttention layer.

        Args:
            query (torch.Tensor): Query embeddings.
            key (torch.Tensor): Key embeddings.
            value (torch.Tensor): Value embeddings.
            **kwargs: Additional keyword arguments passed to the nn.MultiheadAttention forward method.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention with LoRA.
        """
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

            set_param(self, 'in_proj_weight', param=p_new, mode="update")
            result = nn.MultiheadAttention.forward(
                self, query, key, value, **kwargs
            )
            set_param(self, 'in_proj_weight', param=p, mode="update")
            
            return result
        
        return nn.MultiheadAttention.forward(self, query, key, value, **kwargs)
