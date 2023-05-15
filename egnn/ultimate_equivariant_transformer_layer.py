from typing import Callable, Optional

import torch
import torch.nn as nn

from egnn.ultimate_equivariant_multihead_attention import EquivariantMultiheadAttention
from egnn.ultimate_utils import *


class EquivariantTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        attention_activation_fn: str = 'softmax',
        num_rbf: float = 128,
        distance_influence: str = 'both',
        distance_activation_fn = nn.SiLU(),
        use_x_layernorm: bool = False,
        use_dx_laynorm: bool = False,
        use_dx_droppath_prob: float = 0.0,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        distance_projection = 'exp',
        trainable_dist_proj = True,
        use_2d = False,
        out_layer: bool = False,
    ) -> None:
        super().__init__()


        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.dx_dropout_module = None
        if use_dx_droppath_prob > 0.0:
            self.dx_dropout_module = DropPath(use_dx_droppath_prob)


        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim) if use_x_layernorm else None
        
        self.dx_ln = nn.LayerNorm(self.embedding_dim) if use_dx_laynorm else None
        
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            attention_activation=attention_activation_fn,
            num_rbf=num_rbf,
            distance_influence=distance_influence,
            distance_activation_fn=distance_activation_fn,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            distance_projection = distance_projection,
            trainable_dist_proj = trainable_dist_proj,
            use_2d = use_2d,
            out_layer=out_layer,
        )

                
            
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        attention_activation,
        num_rbf,
        distance_influence,
        distance_activation_fn,
        cutoff_lower,
        cutoff_upper,
        distance_projection,
        trainable_dist_proj,
        use_2d,
        out_layer,
    ):
        return EquivariantMultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            attention_activation=attention_activation,
            num_rbf=num_rbf,
            distance_influence=distance_influence,
            distance_activation_fn=distance_activation_fn,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            distance_projection = distance_projection,
            trainable = trainable_dist_proj,
            use_2d = use_2d,
            out_layer=out_layer,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        velocity: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
#         edge_index: Optional[torch.Tensor] = None,
#         edge_feature: Optional[torch.Tensor] = None,
#         edge_direction: Optional[torch.Tensor] = None,
#         cutoff: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
                 
        x = self.self_attn_layer_norm(x) if self.self_attn_layer_norm is not None else x
            
        ###########################################################################################################
        dx, dvec, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            pos=pos,
            velocity=velocity,
            adj=adj,
#             edge_index=edge_index,
#             edge_feature=edge_feature,
#             edge_direction=edge_direction,
#             cutoff=cutoff,
            node_mask=node_mask,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        if self.dx_dropout_module is not None:
            dx = self.dx_dropout_module(dx)
            
            
        if self.dx_ln:
            dx = self.dx_ln(dx)
            

        return dx, dvec, attn