import math
from typing import Optional, Tuple

import torch
from egnn.ultimate_utils import *
from torch import Tensor, nn


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        edge_embed_dim,
        graph_embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        use_adjacency=False,
        use_graph_embedding=False,
        use_graph_embedding_bias=False,
        attention_activation='softmax',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.use_adjacency = use_adjacency
        self.use_graph_embedding = use_graph_embedding
        self.use_graph_embedding_bias = use_graph_embedding_bias
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

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        
        
        if self.use_adjacency:
            self.e_add_proj = quant_noise(
                nn.Linear(edge_embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

            self.e_mul_proj = quant_noise(
                nn.Linear(edge_embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
            
            self.e_out_proj = quant_noise(
            nn.Linear(embed_dim, edge_embed_dim, bias=bias), q_noise, qn_block_size
            )
            
            
        if self.use_graph_embedding:
            self.g_add_proj = quant_noise(
                nn.Linear(graph_embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

            self.g_mul_proj = quant_noise(
                nn.Linear(graph_embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
            
            self.g_out_proj = quant_noise(
                nn.Linear(graph_embed_dim, graph_embed_dim, bias=bias), q_noise, qn_block_size
            )
            
            self.node_to_graph_proj = Node2GraphLayer(embed_dim, graph_embed_dim)
            
            if self.use_adjacency:
                self.edge_to_graph_proj = Edge2GraphLayer(edge_embed_dim, graph_embed_dim)
            
            if self.use_graph_embedding_bias:
                self.g_bias_proj = quant_noise(
                nn.Linear(graph_embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
                )
        
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

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
            
        if self.use_adjacency:
            nn.init.xavier_uniform_(self.e_add_proj.weight)
            if self.e_add_proj.bias is not None:
                nn.init.constant_(self.e_add_proj.bias, 0.0)
                
            nn.init.xavier_uniform_(self.e_mul_proj.weight)
            if self.e_mul_proj.bias is not None:
                nn.init.constant_(self.e_mul_proj.bias, 0.0)
                
            nn.init.xavier_uniform_(self.e_out_proj.weight)
            if self.e_out_proj.bias is not None:
                nn.init.constant_(self.e_out_proj.bias, 0.0)
                
                
        if self.use_graph_embedding:
            nn.init.xavier_uniform_(self.g_add_proj.weight)
            if self.g_add_proj.bias is not None:
                nn.init.constant_(self.g_add_proj.bias, 0.0)
                
            nn.init.xavier_uniform_(self.g_mul_proj.weight)
            if self.g_mul_proj.bias is not None:
                nn.init.constant_(self.g_mul_proj.bias, 0.0)
                
            nn.init.xavier_uniform_(self.g_out_proj.weight)
            if self.g_out_proj.bias is not None:
                nn.init.constant_(self.g_out_proj.bias, 0.0)
                
            if self.use_graph_embedding_bias:
                nn.init.xavier_uniform_(self.g_bias_proj.weight)
                if self.g_bias_proj.bias is not None:
                    nn.init.constant_(self.g_bias_proj.bias, 0.0)


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        adj: Optional[Tensor] = None,
        graph_feature: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
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
            # q[:, 1:, :] = q[:, 1:, :] * node_mask
            # k[:, 1:, :] = k[:, 1:, :] * node_mask
            # v[:, 1:, :] = v[:, 1:, :] * node_mask
            q = q * node_mask
            k = k * node_mask
            v = v * node_mask
        
        q *= self.scaling

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
                .view(bsz, 1, tgt_len, self.num_heads, self.head_dim)
            )

        assert k is not None
        assert k.size(2) == src_len
                
        
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = q * k
        #attn_weights = attn_weights / math.sqrt(attn_weights.size(-1))
        #attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len, self.num_heads, self.head_dim]
        
        # Compute for adjacency
        if (adj is not None) & self.use_adjacency:
            n_nodes = adj.shape[1]
            E1 = self.e_mul_proj(adj)
            E2 = self.e_add_proj(adj) 

            
            if edge_mask is not None:
                E1 = E1 * edge_mask
                E2 = E2 * edge_mask
            
            E1 = E1.view(bsz, n_nodes, n_nodes, self.num_heads, self.head_dim)
            E2 = E2.view(bsz, n_nodes, n_nodes, self.num_heads, self.head_dim)
            
            # Incorporate edge features to the self attention scores.
            # attn_weights[:, 1:, 1:, :, :] = (attn_weights[:, 1:, 1:, :, :].clone() * (E1 + 1)) + E2
            
            # new_adj = attn_weights[:, 1:, 1:, :, :].view(bsz, n_nodes, n_nodes, self.num_heads*self.head_dim) + attn_weights[:, 0, 0, :, :].view(bsz, 1, 1, self.num_heads*self.head_dim)
            attn_weights = (attn_weights * (E1 + 1)) + E2
            
            new_adj = attn_weights.clone().view(bsz, n_nodes, n_nodes, self.num_heads*self.head_dim)
            
            if (graph_feature is not None) & self.use_graph_embedding:
                G1 = self.g_mul_proj(graph_feature).view(bsz, 1, 1, self.num_heads*self.head_dim)
                G2 = self.g_add_proj(graph_feature).view(bsz, 1, 1, self.num_heads*self.head_dim)
                new_adj = new_adj * (G1 + 1) + G2
            
            new_adj = self.e_out_proj(new_adj) * edge_mask
            
            
        # if (graph_feature is not None) & self.use_graph_embedding & self.use_graph_embedding_bias:
        #     G_bias = self.g_bias_proj(graph_feature).view(bsz, self.num_heads, self.head_dim)
        #     attn_weights[:, 0, 0, :, :] = attn_weights[:, 0, 0, :, :] + G_bias
            
            
        if (graph_feature is not None) & self.use_graph_embedding:
            node2graph = self.node_to_graph_proj(query)
            new_graph_feature = node2graph + graph_feature
            
            if (adj is not None) & self.use_adjacency:
                edge2graph = self.edge_to_graph_proj(adj)
                new_graph_feature = edge2graph + new_graph_feature
                
            new_graph_feature = self.g_out_proj(new_graph_feature)
                    
                        
        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz, tgt_len, src_len, self.num_heads, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbol
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
        

        attn_weights_float = torch.nan_to_num(attn_weights_float)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = attn_probs * v
        attn = attn.sum(dim=2)
        assert list(attn.size()) == [bsz, tgt_len, self.num_heads, self.head_dim]

        attn = attn.contiguous().view(bsz, tgt_len, embed_dim)
        attn = self.out_proj(attn)
        
        if node_mask is not None:
            #attn[:, 1:, :] = attn[:, 1:, :] * node_mask
            attn = attn * node_mask


        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len, self.head_dim
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        if ((adj is not None) & self.use_adjacency) and ((graph_feature is not None) & self.use_graph_embedding):
                return attn, attn_weights, new_adj, new_graph_feature
            
        if not ((adj is not None) & self.use_adjacency) and ((graph_feature is not None) & self.use_graph_embedding):
                return attn, attn_weights, None, new_graph_feature
            
        if ((adj is not None) & self.use_adjacency) and not ((graph_feature is not None) & self.use_graph_embedding):
                return attn, attn_weights, new_adj, None
            
        return attn, attn_weights, None, None

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    
    
class Node2GraphLayer(nn.Module):
    def __init__(self, emb_node, emb_graph):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * emb_node, emb_graph)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, 0.0)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

    

class Edge2GraphLayer(nn.Module):
    def __init__(self, emb_edge, emb_graph):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * emb_edge, emb_graph)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, 0.0)
            
    def forward(self, E):
        """ E: bs, n, n, de """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out