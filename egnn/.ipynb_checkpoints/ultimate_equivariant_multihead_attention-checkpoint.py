import math
from typing import Optional, Tuple

import torch
from egnn.ultimate_utils import *
from torch import Tensor, nn
import torch_geometric
from torch_scatter import scatter


class EquivariantMultiheadAttention(nn.Module):
#class EquivariantMultiheadAttention(torch_geometric.nn.MessagePassing):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        attention_activation='softmax',
        num_rbf=128,
        distance_influence='both',
        distance_activation_fn=nn.SiLU(),
    ):
        super(EquivariantMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.attention_activation = attention_activation

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(p=dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        
        
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        
        
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim * 3, bias=bias), q_noise, qn_block_size
        )
        
        
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim * 3, bias=bias), q_noise, qn_block_size
        )
        
        self.velocity_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        
        
        self.distance_activation_fn = distance_activation_fn
        self.distance_k_proj = None
        if distance_influence in ["keys", "both"]:
            self.distance_k_proj = nn.Linear(num_rbf, embed_dim)

        self.distance_v_proj = None
        if distance_influence in ["values", "both"]:
            self.distance_v_proj = nn.Linear(num_rbf, embed_dim * 3)
            
        self.node_dim = 0
        self.reset_parameters()


    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

            
        if self.distance_k_proj is not None:
            nn.init.xavier_uniform_(self.distance_k_proj.weight)
            if self.distance_k_proj.bias is not None:
                nn.init.constant_(self.distance_k_proj.bias, 0.0)
                
                
        if self.distance_v_proj is not None:
            nn.init.xavier_uniform_(self.distance_v_proj.weight)
            if self.distance_v_proj.bias is not None:
                nn.init.constant_(self.distance_v_proj.bias, 0.0)
            
            
        nn.init.xavier_uniform_(self.velocity_proj.weight)
        if self.velocity_proj.bias is not None:
            nn.init.constant_(self.velocity_proj.bias, 0.0)
            
            
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
            
                

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        velocity: Optional[Tensor],
        attn_bias: Optional[Tensor],
        edge_index: Optional[Tensor] = None,
        edge_feature: Optional[Tensor] = None,
        edge_direction: Optional[Tensor] = None,
        cutoff: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [bsz, tgt_len, embed_dim]
        if key is not None:
            key_bsz, src_len, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        
        if node_mask is not None:
            q = q * node_mask
            k = k * node_mask
            v = v * node_mask
        
        q *= self.scaling
        
        if v is not None:
            v = (
                v.contiguous()
                .view(bsz, tgt_len, self.num_heads, self.head_dim * 3)
            )
            
            
        distance_k_proj = (
            self.distance_activation_fn(self.distance_k_proj(edge_feature)).view(bsz, tgt_len, tgt_len, self.num_heads, self.head_dim)
            if self.distance_k_proj is not None
            else None
        )
        
        
        distance_v_proj = (
            self.distance_activation_fn(self.distance_v_proj(edge_feature)).view(bsz, tgt_len, tgt_len, self.num_heads, self.head_dim*3)
            if self.distance_v_proj is not None
            else None
        )

        q = (
            q.contiguous()
            .view(bsz, tgt_len, 1, self.num_heads, self.head_dim)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(bsz, 1, tgt_len, self.num_heads, self.head_dim)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(bsz, 1, tgt_len, self.num_heads, self.head_dim*3)
            )

        assert k is not None
        assert k.size(2) == src_len
        
        #v = v.view(bsz, tgt_len, 1, self.num_heads, self.head_dim*3) * v.view(bsz, 1, tgt_len, self.num_heads, self.head_dim*3)
        
        v = (v * distance_v_proj) if distance_v_proj is not None else v
            
        v, value_proj1, value_proj2 = torch.split(v, self.head_dim, dim=-1)
        
        
        
        # compute velocity
        velocity_proj1, velocity_proj2, velocity_proj3 = torch.split(self.velocity_proj(velocity), embed_dim, dim=-1)
        velocity = (
                velocity.contiguous()
                .view(bsz, tgt_len, 3, self.num_heads, self.head_dim)
            )
        velocity_dot = (velocity_proj1 * velocity_proj2).sum(2)
                

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = q * k * distance_k_proj if distance_k_proj is not None else q * k
        
        attn_weights = attn_weights / math.sqrt(attn_weights.size(-1))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len, self.num_heads, self.head_dim]
        

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).to(torch.bool),
                float(-1e-20),
            )

        if before_softmax:
            if (adj is not None) & self.use_adjacency:
                return attn_weights, v, new_adj
            
            return attn_weights, v, None

        if self.attention_activation in {'softmax'}:
            attn_weights_float = nn.functional.softmax(
                attn_weights, dim=2
            )
        elif self.attention_activation in {'silu'}:
            attn_weights_float = nn.functional.silu(
                attn_weights
            )
        elif self.attention_activation in {'gelu'}:
            attn_weights_float = nn.functional.gelu(
                attn_weights
            )
        elif self.attention_activation in {'relu'}:
            attn_weights_float = nn.functional.relu(
                attn_weights
            )
    
        cutoff = cutoff.unsqueeze(-1).unsqueeze(-1)
        
        attn_weights_float = torch.nan_to_num(attn_weights_float)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_weights = attn_weights * cutoff
        
        attn_probs = self.dropout_module(attn_weights)
        
        assert v is not None
        attn = attn_probs * v * (cutoff != 0)
        attn = attn.sum(2)
        assert list(attn.size()) == [bsz, tgt_len, self.num_heads, self.head_dim]

        attn = attn.contiguous().view(bsz, tgt_len, embed_dim)
        o1, o2, o3 = torch.split(self.out_proj(attn), embed_dim, dim=-1)
            
            

        vec1 = velocity.unsqueeze(1) * value_proj1.unsqueeze(3)
        vec2 = value_proj2.unsqueeze(3) * (edge_direction.unsqueeze(-1).unsqueeze(-1))
        vec = ((vec1 + vec2).view(bsz, tgt_len, tgt_len, 3, embed_dim) * (cutoff != 0)).sum(2)

        
        dx = velocity_dot * o2 + o3
        dvec = velocity_proj3 * o1.unsqueeze(2) + vec
        
        if node_mask is not None:
            dx = dx * node_mask
            dvec = dvec * node_mask.unsqueeze(-1)
        
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len, self.head_dim
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
                
            return dx, dvec, attn_weights
                
        return dx, dvec, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
        
#         distance_k_proj_hat = (distance_k_proj*cutoff.unsqueeze(-1).unsqueeze(-1))
#         distance_v_proj_hat = (distance_v_proj*cutoff.unsqueeze(-1).unsqueeze(-1))
#         edge_direction_hat = (edge_direction*cutoff.unsqueeze(-1))

        
#         distance_k_proj = (distance_k_proj[distance_k_proj_hat != 0]).view(-1, self.num_heads, self.head_dim)
#         distance_v_proj = (distance_v_proj[distance_v_proj_hat != 0]).view(-1, self.num_heads, self.head_dim*3)
#         edge_direction = (edge_direction[edge_direction_hat != 0]).view(-1, 3)
#         velocity = (velocity).view(-1, 3*self.num_heads*self.head_dim)
#         cutoff = cutoff[cutoff!=0].view(-1)
#         q = q.view(-1, self.num_heads*self.head_dim)
#         k = k.view(-1, self.num_heads*self.head_dim)
#         v = v.view(-1, self.num_heads*self.head_dim*3)

        
#         x, vec = self.propagate(
#             edge_index,
#             q=q,
#             k=k,
#             v=v,
#             vec=velocity,
#             dk=distance_k_proj,
#             dv=distance_v_proj,
#             r_ij=cutoff,
#             d_ij=edge_direction,
#             size=None,
#         )
#         x = x.reshape(bsz, tgt_len, embed_dim)
#         vec = vec.reshape(bsz, tgt_len, 3, embed_dim)

#         o1, o2, o3 = torch.split(self.out_proj(x), embed_dim, dim=-1)
#         dx = velocity_dot * o2 + o3
#         dvec = velocity_proj3 * o1.unsqueeze(2) + vec

#         print('in', dx[0, :, 0])
#         return dx, dvec, None

#     def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
#         q_i = q_i.view(-1, self.num_heads, self.head_dim)
#         k_j = k_j.view(-1, self.num_heads, self.head_dim)
#         v_j = v_j.view(-1, self.num_heads, self.head_dim*3)
#         vec_j = vec_j.view(-1, 3, self.num_heads, self.head_dim) 
#         # attention mechanism
#         if dk is None:
#             attn = (q_i * k_j).sum(dim=-1)
#         else:
#             attn = (q_i * k_j * dk).sum(dim=-1)

#         # attention activation function
#         attn = (attn) * r_ij.unsqueeze(1)

#         # value pathway
#         if dv is not None:
#             v_j = v_j * dv
#         x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=-1)

#         # update scalar features
#         # update vector features
#         vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
#         return x, vec

#     def aggregate(
#         self,
#         features: Tuple[torch.Tensor, torch.Tensor],
#         index: torch.Tensor,
#         ptr: Optional[torch.Tensor],
#         dim_size: Optional[int],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         x, vec = features
#         x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
#         vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
#         return x, vec

#     def update(
#         self, inputs: Tuple[torch.Tensor, torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         return inputs
    
