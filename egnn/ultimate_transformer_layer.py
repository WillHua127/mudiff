from typing import Callable, Optional

import torch
import torch.nn as nn

from egnn.ultimate_multihead_attention import MultiheadAttention
from egnn.ultimate_utils import *


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 768,
        edge_embedding_dim: int = 768,
        graph_embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        ffn_edge_embedding_dim: int = 3072,
        ffn_graph_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_dropout_adj: float = 0.1,
        activation_dropout_graph_feature: float = 0.1,
        activation_fn = nn.SiLU(),
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.1,
        droppath_adj_prob: float = 0.1,
        droppath_graph_feature_prob: float = 0.1,
        use_adjacency: bool = False,
        use_graph_embedding: bool = False,
        use_graph_embedding_bias: bool = False,
        attention_activation_fn: str = 'softmax'
    ) -> None:
        super().__init__()


        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.use_adjacency = use_adjacency
        self.edge_embedding_dim = edge_embedding_dim
        self.use_graph_embedding = use_graph_embedding
        self.use_graph_embedding_bias = use_graph_embedding_bias
        self.graph_embedding_dim = graph_embedding_dim

        self.dropout_module = None
        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
            
        self.dropout_adj_module = None
        if (droppath_adj_prob > 0.0) and self.use_adjacency:
            self.dropout_adj_module = DropPath(droppath_adj_prob)
            
        self.dropout_graph_feature_module = None
        if (droppath_graph_feature_prob > 0.0) and self.use_graph_embedding:
            self.dropout_graph_feature_module = DropPath(droppath_graph_feature_prob)
            
            
        self.dropout_activation = None
        if activation_dropout > 0.0:
            self.dropout_activation = nn.Dropout(p=activation_dropout)
            
            
        self.dropout_adj_activation = None
        if (activation_dropout_adj > 0.0) and self.use_adjacency:
            self.dropout_adj_activation = nn.Dropout(p=activation_dropout_adj)
            
            
        self.dropout_graph_feature_activation = None
        if (activation_dropout_graph_feature > 0.0) and self.use_graph_embedding:
            self.dropout_graph_feature_activation = nn.Dropout(p=activation_dropout_graph_feature)
            

        # Initialize blocks
        self.activation_fn = activation_fn
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            edge_embedding_dim,
            graph_embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            use_adjacency=use_adjacency,
            use_graph_embedding=use_graph_embedding,
            use_graph_embedding_bias=use_graph_embedding_bias,
            attention_activation=attention_activation_fn,
        )

        self.sandwich_ln = sandwich_ln

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        # layer norm associated with the self attention layer, sandwich
        self.self_attn_sandwich_layer_norm = nn.LayerNorm(self.embedding_dim) if self.sandwich_ln else None
            

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.final_sandwich_layer_norm = nn.LayerNorm(self.embedding_dim) if self.sandwich_ln else None
        
        
        if self.use_adjacency:
            self.self_edge_attn_layer_norm = nn.LayerNorm(self.edge_embedding_dim)
            self.self_edge_attn_sandwich_layer_norm = nn.LayerNorm(self.edge_embedding_dim) if self.sandwich_ln else None
            self.final_edge_sandwich_layer_norm = nn.LayerNorm(self.edge_embedding_dim) if self.sandwich_ln else None
            self.final_edge_layer_norm = nn.LayerNorm(self.edge_embedding_dim)
            
            self.fc1_adj = self.build_fc1(
                self.edge_embedding_dim,
                ffn_edge_embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            self.fc2_adj = self.build_fc2(
                ffn_edge_embedding_dim,
                self.edge_embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

            
            
        if self.use_graph_embedding:
            self.self_graph_attn_layer_norm = nn.LayerNorm(self.graph_embedding_dim)
            self.self_graph_attn_sandwich_layer_norm = nn.LayerNorm(self.graph_embedding_dim) if self.sandwich_ln else None
            self.final_graph_sandwich_layer_norm = nn.LayerNorm(self.graph_embedding_dim) if self.sandwich_ln else None
            self.final_graph_layer_norm = nn.LayerNorm(self.graph_embedding_dim)
            
            self.fc1_graph_feature = self.build_fc1(
                self.graph_embedding_dim,
                ffn_graph_embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            self.fc2_graph_feature = self.build_fc2(
                ffn_graph_embedding_dim,
                self.graph_embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

            
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        edge_embed_dim,
        graph_embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        use_adjacency,
        use_graph_embedding,
        use_graph_embedding_bias,
        attention_activation,
    ):
        return MultiheadAttention(
            embed_dim,
            edge_embed_dim,
            graph_embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            use_adjacency=use_adjacency,
            use_graph_embedding=use_graph_embedding,
            use_graph_embedding_bias=use_graph_embedding_bias,
            attention_activation=attention_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        graph_feature: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: B x T x C1
        # adj: B x T x T x C2
        # graph_feature: B x C3
        ###########################################################################################################
        residual = x
        
        if (adj is not None) and (self.use_adjacency):
            residual_e = adj
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            residual_g = graph_feature
            
        
        ###########################################################################################################
        if self.sandwich_ln:
            x = self.self_attn_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.self_edge_attn_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.self_graph_attn_layer_norm(graph_feature)
            
            
        ###########################################################################################################
        x, attn, adj, graph_feature = self.self_attn(
            query=x,
            key=x,
            value=x,
            adj=adj,
            graph_feature=graph_feature,
            node_mask=node_mask,
            edge_mask=edge_mask,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        ###########################################################################################################
        if self.dropout_module is not None:
            x = self.dropout_module(x)
            
        if (adj is not None) and (self.dropout_adj_module is not None):
            adj = self.dropout_adj_module(adj)
            
        if (graph_feature is not None) and (self.dropout_graph_feature_module is not None):
            graph_feature = self.dropout_graph_feature_module(graph_feature)
            
            
        ###########################################################################################################
        if self.sandwich_ln:
            x = self.self_attn_sandwich_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.self_edge_attn_sandwich_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.self_graph_attn_sandwich_layer_norm(graph_feature)
            
            
        ###########################################################################################################
        x = residual + x
        
        if (adj is not None) and (self.use_adjacency):
            adj = residual_e + adj
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            graph_feature = residual_g + graph_feature
            
            
        ###########################################################################################################
        if not self.sandwich_ln:
            x = self.self_attn_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.self_edge_attn_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.self_graph_attn_layer_norm(graph_feature)

   
        ###########################################################################################################
        residual = x
        
        if (adj is not None) and (self.use_adjacency):
            residual_e = adj
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            residual_g = graph_feature
            
            
        ###########################################################################################################
        if self.sandwich_ln:
            x = self.final_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.final_edge_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.final_graph_layer_norm(graph_feature)
            
            
        ###########################################################################################################
        x = self.activation_fn(self.fc1(x))
        
        if (adj is not None) and (self.use_adjacency):
            adj = self.activation_fn(self.fc1_adj(adj))
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            graph_feature = self.activation_fn(self.fc1_graph_feature(graph_feature))
            
        if self.dropout_activation is not None:
            x = self.dropout_activation(x)
            
        if (adj is not None) and (self.dropout_adj_activation is not None):
            adj = self.dropout_adj_activation(adj)
            
        if (graph_feature is not None) and (self.dropout_graph_feature_activation is not None):
            graph_feature = self.dropout_graph_feature_activation(graph_feature)
        
        
        ###########################################################################################################
        x = self.fc2(x)
        
        if (adj is not None) and (self.use_adjacency):
            adj = self.fc2_adj(adj)
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            graph_feature = self.fc2_graph_feature(graph_feature)
            
            
        ###########################################################################################################
        if self.dropout_module is not None:
            x = self.dropout_module(x)
            
        if (adj is not None) and (self.dropout_adj_module is not None):
            adj = self.dropout_adj_module(adj)
            
        if (graph_feature is not None) and (self.dropout_graph_feature_module is not None):
            graph_feature = self.dropout_graph_feature_module(graph_feature)
            
            
        ###########################################################################################################
        if self.sandwich_ln:
            x = self.final_sandwich_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.final_edge_sandwich_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.final_graph_sandwich_layer_norm(graph_feature)
                
                
        ###########################################################################################################
        x = residual + x
        
        if (adj is not None) and (self.use_adjacency):
            adj = residual_e + adj
            
        if (graph_feature is not None) and (self.use_graph_embedding):
            graph_feature = residual_g + graph_feature
            
            
        ###########################################################################################################
        if not self.sandwich_ln:
            x = self.final_layer_norm(x)
            
            if (adj is not None) and (self.use_adjacency):
                adj = self.final_edge_layer_norm(adj)
                
            if (graph_feature is not None) and (self.use_graph_embedding):
                graph_feature = self.final_graph_layer_norm(graph_feature)
                
                
        ###########################################################################################################
        if ((adj is not None) & self.use_adjacency) and ((graph_feature is not None) & self.use_graph_embedding):
            return x, attn, adj, graph_feature
        
        if not ((adj is not None) & self.use_adjacency) and ((graph_feature is not None) & self.use_graph_embedding):
            return x, attn, None, graph_feature
        
        if ((adj is not None) & self.use_adjacency) and not ((graph_feature is not None) & self.use_graph_embedding):
            return x, attn, adj, None
        
        return x, attn, None, None