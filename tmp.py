import torch
import torch.nn as nn

from torch.nn import Module, Parameter
from torch.nn.functional import linear
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import warnings
from typing import Optional, Tuple

# # class MyMultiheadAttention(nn.MultiheadAttention):
# #     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
# #                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
# #         factory_kwargs = {'device': device, 'dtype': dtype}
# #         super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, **factory_kwargs)

# #     @property
# #     def q_proj(self):
# #         if hasattr(self, 'q_proj_weight'):
# #             return nn.Parameter(self.q_proj_weight.clone().detach())
# #         return None

# #     @property
# #     def k_proj(self):
# #         if hasattr(self, 'k_proj_weight'):
# #             return nn.Parameter(self.k_proj_weight.clone().detach())
# #         return None

# #     @property
# #     def v_proj(self):
# #         if hasattr(self, 'v_proj_weight'):
# #             return nn.Parameter(self.v_proj_weight.clone().detach())
# #         return None

# #     def __repr__(self):
# #         return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, q_proj={self.q_proj_weight}, k_proj={self.k_proj_weight}, v_proj={self.v_proj_weight}, out_proj={self.out_proj})"

# # # Create an instance of your custom class
# # my_multihead_attention = MyMultiheadAttention(embed_dim=512, num_heads=8)

# # # Now when you print it, you'll see the custom representation
# # print(my_multihead_attention)
# Tensor = torch.Tensor

# class MultiheadAttention(Module):
    

#     __constants__ = ['batch_first']
#     bias_k: Optional[torch.Tensor]
#     bias_v: Optional[torch.Tensor]

#   # def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#               #  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         if not self._qkv_same_embed_dim:
#             self.q_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
#             self.k_proj = NonDynamicallyQuantizableLinear(embed_dim, self.kdim, bias=bias, **factory_kwargs)
#             self.v_proj = NonDynamicallyQuantizableLinear(embed_dim, self.vdim, bias=bias, **factory_kwargs)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)

#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self._reset_parameters()

#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)

#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True

#         super().__setstate__(state)

#     def forward(
#             self,
#             query: Tensor,
#             key: Tensor,
#             value: Tensor,
#             key_padding_mask: Optional[Tensor] = None,
#             need_weights: bool = True,
#             attn_mask: Optional[Tensor] = None,
#             average_attn_weights: bool = True,
#             is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:

#         is_batched = query.dim() == 3

#         key_padding_mask = F._canonical_mask(
#             mask=key_padding_mask,
#             mask_name="key_padding_mask",
#             other_type=F._none_or_dtype(attn_mask),
#             other_name="attn_mask",
#             target_type=query.dtype
#         )

#         attn_mask = F._canonical_mask(
#             mask=attn_mask,
#             mask_name="attn_mask",
#             other_type=None,
#             other_name="",
#             target_type=query.dtype,
#             check_other=False,
#         )

#         why_not_fast_path = ''
#         if not is_batched:
#             why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
#         elif query is not key or key is not value:
#             # When lifting this restriction, don't forget to either
#             # enforce that the dtypes all match or test cases where
#             # they don't!
#             why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
#         elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
#             why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
#         elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
#             # this case will fail anyway, but at least they'll get a useful error message.
#             why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
#         elif self.training:
#             why_not_fast_path = "training is enabled"
#         elif not self.batch_first:
#             why_not_fast_path = "batch_first was not True"
#         elif self.bias_k is not None:
#             why_not_fast_path = "self.bias_k was not None"
#         elif self.bias_v is not None:
#             why_not_fast_path = "self.bias_v was not None"
#         elif self.add_zero_attn:
#             why_not_fast_path = "add_zero_attn was enabled"
#         elif not self._qkv_same_embed_dim:
#             why_not_fast_path = "_qkv_same_embed_dim was not True"
#         elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
#             why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
#                                  is not supported with NestedTensor input"
#         elif torch.is_autocast_enabled():
#             why_not_fast_path = "autocast is enabled"

#         if not why_not_fast_path:
#             tensor_args = (
#                 query,
#                 key,
#                 value,
#                 self.in_proj_weight,
#                 self.in_proj_bias,
#                 self.out_proj.weight,
#                 self.out_proj.bias,
#             )
#             # We have to use list comprehensions below because TorchScript does not support
#             # generator expressions.
#             if torch.overrides.has_torch_function(tensor_args):
#                 why_not_fast_path = "some Tensor argument has_torch_function"
#             elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
#                 why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
#             elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
#                 why_not_fast_path = ("grad is enabled and at least one of query or the "
#                                      "input/output projection weights or biases requires_grad")
#             if not why_not_fast_path:
#                 merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

#                 return torch._native_multi_head_attention(
#                     query,
#                     key,
#                     value,
#                     self.embed_dim,
#                     self.num_heads,
#                     self.in_proj_weight,
#                     self.in_proj_bias,
#                     self.out_proj.weight,
#                     self.out_proj.bias,
#                     merged_mask,
#                     need_weights,
#                     average_attn_weights,
#                     mask_type)

#         any_nested = query.is_nested or key.is_nested or value.is_nested
#         assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
#                                 f"The fast path was not hit because {why_not_fast_path}")

#         if self.batch_first and is_batched:
#             # make sure that the transpose op does not affect the "is" property
#             if key is value:
#                 if query is key:
#                     query = key = value = query.transpose(1, 0)
#                 else:
#                     query, key = [x.transpose(1, 0) for x in (query, key)]
#                     value = key
#             else:
#                 query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask,
#                 use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight,
#                 average_attn_weights=average_attn_weights,
#                 is_causal=is_causal)
#         else:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask,
#                 need_weights=need_weights,
#                 attn_mask=attn_mask,
#                 average_attn_weights=average_attn_weights,
#                 is_causal=is_causal)
#         if self.batch_first and is_batched:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights

#     def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
#                     query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
 
#         mask_type: Optional[int] = None
#         merged_mask: Optional[Tensor] = None

#         attn_mask = F._canonical_mask(
#             mask=attn_mask,
#             mask_name="attn_mask",
#             other_type=None,
#             other_name="",
#             target_type=query.dtype,
#             check_other=False,
#         )

#         if key_padding_mask is not None:
#             mask_type = 1
#             merged_mask = key_padding_mask

#         if attn_mask is not None:
#             # In this branch query can't be a nested tensor, so it has a shape
#             batch_size, seq_len, _ = query.shape
#             mask_type = 2

#             # Always expands attn_mask to 4D
#             if attn_mask.dim() == 3:
#                 attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
#             else:  # attn_mask.dim() == 2:
#                 attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
#             merged_mask = attn_mask_expanded

#             if key_padding_mask is not None:
#                 key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
#                 merged_mask = attn_mask_expanded + key_padding_mask_expanded

#         # no attn_mask and no key_padding_mask, returns None, None
#         return merged_mask, mask_type
    
    
# multihead_attn = MultiheadAttention(embed_dim=768, num_heads=12)
# attn_output, attn_output_weights = multihead_attn(query, key, value)

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch import Tensor
from typing import Optional, Tuple

class MyMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, **factory_kwargs)
        self.register_parameter('q_proj', self.q_proj_weight)
        self.register_parameter('k_proj', self.k_proj_weight)
        self.register_parameter('v_proj', self.v_proj_weight)
        
 
    def __repr__(self):
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, q_proj={self.q_proj_weight}, k_proj={self.k_proj_weight}, v_proj={self.v_proj_weight}, out_proj={self.out_proj})"
multihead_attn = MyMultiheadAttention(embed_dim=768, num_heads=12)
print(multihead_attn)
# attn_output, attn_output_weights = multihead_attn(query, key, value)