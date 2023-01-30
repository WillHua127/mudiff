from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from egnn.ultimate_embedding_diffusion import *
from egnn.ultimate_utils import *
from egnn.ultimate_transformer_layer import *
from egnn.ultimate_equivariant_transformer_layer import *
from egnn.ultimate_equivariant_output import *


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        in_node_dim: int,
        in_edge_dim: int,
        edge_type: str,
        multi_hop_max_dist: int,
        max_weight: int = 0.0,
        atom_weights: list = None,
        max_n_nodes: int = 30,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        edge_embedding_dim: int = 300,
        num_attention_heads: int = 8,
        use_3d_embedding: bool = False,
        num_3d_bias_kernel: int = 128,
        use_2d_embedding: bool = False,
        use_2d_neighbor_embedding: bool = False,
        use_3d_neighbor_embedding: bool = False,
        apply_concrete_adjacency_neighbor: bool = False,
        use_2d_edge_embedding: bool = False,
        cutoff_upper: float = 5.0,
        cutoff_lower: float = 0.0,
        distance_projection: str = 'exp',
        trainable_dist_proj: bool = False,
        neighbor_combine_embedding: str = 'cat',
        use_extra_graph_embedding: bool = False,
        use_extra_graph_embedding_attn_bias: bool = False,
        extra_feature_type: str = 'all', 
        graph_embedding_dim: int = 300,
        ### transformer
        before_attention_dropout: float = 0.0,
        before_attention_layernorm: bool = False,
        before_attention_quant_noise: float = 0.0,
        before_attention_qn_block_size: int = 8,
        ffn_embedding_dim: int = 3072,
        ffn_edge_embedding_dim: int = 3072,
        ffn_graph_embedding_dim: int = 3072,
        in_attention_feature_dropout: float = 0.0,
        in_attention_dropout: float = 0.0,
        in_attention_activation_dropout: float = 0.0,
        in_attention_activation_dropout_adj: float = 0.0,
        in_attention_activation_dropout_graph_feature: float = 0.0,
        in_attention_activation_fn = nn.SiLU(),
        in_attention_quant_noise: float = 0.0,
        in_attention_qn_block_size: int = 8,
        in_attention_layernorm: bool = False,
        in_attention_droppath: float = 0.0,
        in_attention_droppath_adj: float = 0.0,
        in_attention_droppath_graph_feature: float = 0.0,
        in_attention_pred_adjacency: bool = False,
        attention_activation_fn: str = 'softmax',
        ### equivariant transformer
        use_equivariant_transformer: bool = False,
        equivariant_distance_influence: str = 'both',
        equivariant_distance_activation_fn = nn.SiLU(),
        equivariant_attention_activation_fn: str = 'softmax',
        equivariant_in_attention_dropout: float = 0.0,
        equivariant_use_x_layernorm: bool = False,
        equivariant_use_dx_layernorm: bool = False,
        equivariant_dx_dropout: float = 0.0,
        equivariant_coord_activation_fn = nn.SiLU(),
        equivariant_apply_concrete_adjacency: float = False,
        ### output
        combine_transformer_output: str = 'e2e',
        combine_transformer_activation_fn = nn.SiLU(),
        use_output_projection: bool = False,
        output_activation_fn = nn.SiLU(),
        out_node_dim: int = 32,
        out_edge_dim: int = 32,
        use_equivariant_output_projection = False,
        equivariant_output_activation_fn = nn.SiLU(),
    ) -> None:

        super().__init__()


        self.atom_feature = AtomEmbedding(
            in_node_dim=in_node_dim,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            use_2d=use_2d_embedding,
        )
        
        self.embedding_dim = embedding_dim
        
        
        self.edge_feature = EdgeEmbedding(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_dim=embedding_dim,
            edge_hidden_dim=edge_embedding_dim,
            n_layers=num_encoder_layers,
        ) if (use_2d_edge_embedding & use_2d_embedding) else None
        
        
        self.graph_embedding = ExtraGraphFeatureEmebedding(
            max_n_nodes=max_n_nodes,
            max_weight=max_weight,
            atom_weights=atom_weights,
            extra_feature_type=extra_feature_type,
            graph_hidden_dim=graph_embedding_dim,
            n_layers=num_encoder_layers,
            use_2d=use_2d_embedding,
        ) if use_extra_graph_embedding else None
        

        self.molecule_attn_bias = MoleculeAttnBias(
            num_heads=num_attention_heads,
            in_edge_dim=in_edge_dim,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            use_2d=use_2d_embedding,
        )


        self.molecule_3d_bias = Molecule3DBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            distance_projection = distance_projection,
            trainable = trainable_dist_proj,
            no_share_rpe=False,
        ) if use_3d_embedding else None
        
        
        self.molecule_2d_neighbor_embedding = MoleculeNeighbor2DEmbedding(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            n_layers=num_encoder_layers,
            hidden_dim=embedding_dim,
        ) if (use_2d_embedding & use_2d_neighbor_embedding) else None
        
        
        self.molecule_3d_neighbor_embedding = MoleculeNeighbor3DEmbedding(
            in_node_dim=in_node_dim,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            num_kernel=num_3d_bias_kernel,
            hidden_dim=embedding_dim,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            distance_projection = distance_projection,
            trainable = trainable_dist_proj,
            use_2d = apply_concrete_adjacency_neighbor,
        ) if (use_3d_embedding & use_3d_neighbor_embedding) else None
        
        
        
        self.molecule_combine_embedding = CombineEmbedding(
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            use_3d = (use_3d_embedding & use_3d_neighbor_embedding),
            use_2d = (use_2d_embedding & use_2d_neighbor_embedding),
            neighbor_combine_embedding = neighbor_combine_embedding,
        ) if (self.atom_feature is not None) or (self.molecule_3d_neighbor_embedding is not None) or (self.molecule_2d_neighbor_embedding is not None) else None
        
        
        
        self.before_attention_layernorm = before_attention_layernorm
        if self.before_attention_layernorm:
            self.before_layernorm = nn.LayerNorm(embedding_dim)
            
            
        self.before_attention_dropout = True if before_attention_dropout > 0.0 else False
        if self.before_attention_dropout:
            self.before_dropout = nn.Dropout(p=before_attention_dropout)
            
        
        self.before_attention_quant_noise = True if before_attention_quant_noise > 0.0 else False
        if self.before_attention_quant_noise:
            self.before_quant_noise = quant_noise(
                nn.Linear(embedding_dim, embedding_dim, bias=False),
                before_attention_quant_noise,
                before_attention_qn_block_size,
            )

            
        self.transformer_2d_layers = nn.ModuleList([])
        
        self.transformer_3d_layers = nn.ModuleList([]) if use_equivariant_transformer else None
        
        self.transformer_2d_layers.extend(
            [
                self.build_transformer_2d_encoder_layer(
                        embedding_dim=embedding_dim,
                        edge_embedding_dim=edge_embedding_dim,
                        graph_embedding_dim=graph_embedding_dim,
                        ffn_embedding_dim=ffn_embedding_dim,
                        ffn_edge_embedding_dim=ffn_edge_embedding_dim,
                        ffn_graph_embedding_dim=ffn_graph_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=in_attention_feature_dropout,
                        attention_dropout=in_attention_dropout,
                        activation_dropout=in_attention_activation_dropout,
                        activation_dropout_adj=in_attention_activation_dropout_adj,
                        activation_dropout_graph_feature=in_attention_activation_dropout_graph_feature,
                        activation_fn=in_attention_activation_fn,
                        q_noise=in_attention_quant_noise,
                        qn_block_size=in_attention_qn_block_size,
                        sandwich_ln = in_attention_layernorm,
                        droppath_prob=in_attention_droppath,
                        droppath_adj_prob=in_attention_droppath_adj,
                        droppath_graph_feature_prob=in_attention_droppath_graph_feature,
                        use_adjacency=in_attention_pred_adjacency,
                        use_graph_embedding=use_extra_graph_embedding,
                        use_graph_embedding_bias=use_extra_graph_embedding_attn_bias,
                        attention_activation_fn=attention_activation_fn,
                        )
                for _ in range(num_encoder_layers)
            ]
        )
        
        
        
        self.transformer_3d_layers.extend(
            [
                self.build_transformer_3d_encoder_layer(
                        embedding_dim=embedding_dim,
                        num_attention_heads=num_attention_heads,
                        attention_dropout=equivariant_in_attention_dropout,
                        q_noise=in_attention_quant_noise,
                        qn_block_size=in_attention_qn_block_size,
                        attention_activation_fn=equivariant_attention_activation_fn,
                        num_rbf=num_3d_bias_kernel,
                        distance_influence=equivariant_distance_influence,
                        distance_activation_fn=equivariant_distance_activation_fn,
                        use_x_layernorm=equivariant_use_x_layernorm,
                        use_dx_laynorm=equivariant_use_dx_layernorm,
                        use_dx_droppath_prob=equivariant_dx_dropout,
                        )
                for _ in range(num_encoder_layers)
            ]
        ) if self.transformer_3d_layers is not None else None
        
        self.num_layers = num_encoder_layers
        
        self.geom_info = GeometricInformation(
                                n_layers=num_encoder_layers,
                                num_edges=num_edges,
                                num_kernel=num_3d_bias_kernel,
                                cutoff_lower=cutoff_lower,
                                cutoff_upper=cutoff_upper,
                                distance_projection = distance_projection,
                                trainable = trainable_dist_proj,
                                use_2d = equivariant_apply_concrete_adjacency,
                            ) if (self.transformer_3d_layers is not None) else None
        
        
        
        self.equivariant_output = EquivariantScalar(
                                            embedding_dim, 
                                            x_out=out_node_dim,
                                            v_out=1, 
                                            activation=equivariant_output_activation_fn
                                    ) if use_equivariant_output_projection else None
            
            
        self.coordinate_proj = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    equivariant_coord_activation_fn,
                                    nn.Linear(embedding_dim, 1),
                                ) if (self.transformer_3d_layers is not None) else None
        
        
        self.out_node_proj = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    output_activation_fn,
                                    nn.LayerNorm(embedding_dim),
                                    nn.Linear(embedding_dim, out_node_dim),
                                ) if use_output_projection else None
        
        
        self.out_edge_proj = nn.Sequential(
                                    nn.Linear(edge_embedding_dim, edge_embedding_dim),
                                    output_activation_fn,
                                    nn.LayerNorm(edge_embedding_dim),
                                    nn.Linear(edge_embedding_dim, out_edge_dim),
                                ) if (use_output_projection and (self.edge_feature is not None)) else None
        
        
        self.combine_transformer_output = combine_transformer_output
        
        
        self.combine_proj = nn.Sequential(
                                    nn.Linear(embedding_dim * 2, embedding_dim),
                                    combine_transformer_activation_fn,
                                    nn.LayerNorm(embedding_dim),
                                ) if (combine_transformer_output in {'cat', 'cat_last'}) and (self.transformer_3d_layers is not None) else None
        
        
    def build_transformer_2d_encoder_layer(
            self,
            embedding_dim,
            edge_embedding_dim,
            graph_embedding_dim,
            ffn_embedding_dim,
            ffn_edge_embedding_dim,
            ffn_graph_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_dropout_adj,
            activation_dropout_graph_feature,
            activation_fn,
            q_noise,
            qn_block_size,
            sandwich_ln,
            droppath_prob,
            droppath_adj_prob,
            droppath_graph_feature_prob,
            use_adjacency,
            use_graph_embedding,
            use_graph_embedding_bias,
            attention_activation_fn
            ):
            
            return TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    edge_embedding_dim=edge_embedding_dim,
                    graph_embedding_dim=graph_embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    ffn_edge_embedding_dim=ffn_edge_embedding_dim,
                    ffn_graph_embedding_dim=ffn_graph_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_dropout_adj=activation_dropout_adj,
                    activation_dropout_graph_feature=activation_dropout_graph_feature,
                    activation_fn=activation_fn,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    sandwich_ln = sandwich_ln,
                    droppath_prob=droppath_prob,
                    droppath_adj_prob=droppath_adj_prob,
                    droppath_graph_feature_prob=droppath_graph_feature_prob,
                    use_adjacency=use_adjacency,
                    use_graph_embedding=use_graph_embedding,
                    use_graph_embedding_bias=use_graph_embedding_bias,
                    attention_activation_fn=attention_activation_fn,
                )
        
        
        
    def build_transformer_3d_encoder_layer(
            self,
            embedding_dim,
            num_attention_heads,
            attention_dropout,
            q_noise,
            qn_block_size,
            attention_activation_fn,
            num_rbf,
            distance_influence,
            distance_activation_fn,
            use_x_layernorm,
            use_dx_laynorm,
            use_dx_droppath_prob,
            ):
            
            return EquivariantTransformerEncoderLayer(
                        embedding_dim=embedding_dim,
                        num_attention_heads=num_attention_heads,
                        attention_dropout=attention_dropout,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        attention_activation_fn=attention_activation_fn,
                        num_rbf=num_rbf,
                        distance_influence=distance_influence,
                        distance_activation_fn=distance_activation_fn,
                        use_x_layernorm=use_x_layernorm,
                        use_dx_laynorm=use_dx_laynorm,
                        use_dx_droppath_prob=use_dx_droppath_prob,
                        )

        
    def forward(
        self,
        node_features, attn_bias, spatial_pos=None, edge_input=None, edge_type=None, pos=None, adj=None, node_mask=None, edge_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        bs, n_nodes = node_features.size()[:2]
        
        ### Preparing padding mask
        padding_mask = (node_mask==0).squeeze(-1) # B x T x 1
        #padding_mask_cls = torch.zeros(bs, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        #padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        
        
        attn_mask = None
        
        ### A.0 Encoding
        
        ### A.1 Encoding (with 2d information)
        x = self.atom_feature(node_features, adj=adj)
        
        
        ### A.2 Obtaining (2d) attention biases
        attn_bias = self.molecule_attn_bias(node_features, attn_bias, spatial_pos, edge_input, edge_type)

        ### A.3 Obtaining (3d) attention biases & Encoding (with 3d information)
        delta_pos = None
        if (self.molecule_3d_bias is not None) and not (pos == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(node_features, pos)

            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            #x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

            
        ### A.4 Encoding (with 2d neighborhood information)
        x_neighbors_2d = None
        if (self.molecule_2d_neighbor_embedding is not None) and (adj is not None):
            x_neighbors_2d = self.molecule_2d_neighbor_embedding(node_features, adj, edge_type)

            
        ### A.5 Encoding (with 3d neighborhood information)
        x_neighbors_3d = None
        if (self.molecule_3d_neighbor_embedding is not None) and not (pos == 0).all():
            x_neighbors_3d = self.molecule_3d_neighbor_embedding(node_features, pos, adj)
            
            
        ### A.6 Combining x with 2d, 3d neighborhood information
        if self.molecule_combine_embedding is not None:
            x = self.molecule_combine_embedding(x, x_neighbors_2d, x_neighbors_3d)
            
            
        ### A.7 Encoding (2d) edge
        encoded_adj = None
        if (self.edge_feature is not None) and (adj is not None):
            encoded_adj = self.edge_feature(node_features, adj, edge_type)
        
        
        ### A.8 Encoding (2d) graph
        graph_feature = None    
        if self.graph_embedding is not None:
            graph_feature = self.graph_embedding(node_features, adj, edge_type, node_mask)
                        
            
        if self.before_attention_quant_noise:
            x = self.before_quant_noise(x)
            
            
        if self.before_attention_layernorm:
            x = self.before_layernorm(x)
            
            
        if self.before_attention_dropout:
            x = self.before_dropout(x)
            
        if node_mask is not None:
            #x[:, 1:, :] = x[:, 1:, :] * node_mask
            x = x[:, 1:, :] * node_mask
            
        if (edge_mask is not None) and (encoded_adj is not None):
            encoded_adj = encoded_adj * edge_mask
                    
        if attn_bias is not None:
            attn_bias = attn_bias.contiguous().permute(0, 2, 3, 1)[:, 1:, 1:, :]
            #attn_bias = attn_bias.contiguous().permute(0, 2, 3, 1)

            
        edge_index_3d, edge_feature_3d, edge_direction_3d, cutoff_3d, velocity = None, None, None, None, None
        if (self.transformer_3d_layers is not None) and (not (pos == 0).all()):
            edge_index_3d, edge_feature_3d, edge_direction_3d, cutoff_3d = self.geom_info(pos, adj)
            
            velocity = torch.zeros(bs, n_nodes, 3, self.embedding_dim, device=x.device)
            
            
        
        ####################################################################################            
        if self.combine_transformer_output in {'e2e'}:
            
            for i in range(0, self.num_layers):
                x, _, encoded_adj, graph_feature = self.transformer_2d_layers[i](x, adj=encoded_adj, graph_feature=graph_feature, node_mask=node_mask, edge_mask=edge_mask, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)


                if (self.transformer_3d_layers is not None) and (velocity is not None):

                    assert velocity.shape[-1] == x.shape[-1]

                    dx, dvec, _ = self.transformer_3d_layers[i](x, velocity, edge_index=edge_index_3d, edge_feature=edge_feature_3d, edge_direction=edge_direction_3d, cutoff=cutoff_3d, node_mask=node_mask, self_attn_padding_mask=(node_features[:,:,0]==0), self_attn_mask=attn_mask)

                    #x[:, 1:, :] = x[:, 1:, :] + dx
                    x = x + dx
                    velocity = velocity + dvec
                    
                   
                
        ####################################################################################            
        if self.combine_transformer_output in {'add'}:
            
            x_3d = x
            for i in range(0, self.num_layers):
                x, _, encoded_adj, graph_feature = self.transformer_2d_layers[i](x, adj=encoded_adj, graph_feature=graph_feature, node_mask=node_mask, edge_mask=edge_mask, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)


                if (self.transformer_3d_layers is not None) and (velocity is not None):

                    assert velocity.shape[-1] == x.shape[-1]

                    dx, dvec, _ = self.transformer_3d_layers[i](x_3d, velocity, edge_index=edge_index_3d, edge_feature=edge_feature_3d, edge_direction=edge_direction_3d, cutoff=cutoff_3d, node_mask=node_mask, self_attn_padding_mask=(node_features[:,:,0]==0), self_attn_mask=attn_mask)

                    x_3d = x_3d + dx
                    velocity = velocity + dvec
                    
                    x = x + x_3d
                    
                    x_3d = x
                    
                    
        ####################################################################################            
        if self.combine_transformer_output in {'cat'}:
            
            x_3d = x
            for i in range(0, self.num_layers):
                x, _, encoded_adj, graph_feature = self.transformer_2d_layers[i](x, adj=encoded_adj, graph_feature=graph_feature, node_mask=node_mask, edge_mask=edge_mask, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)


                if (self.transformer_3d_layers is not None) and (velocity is not None):

                    assert velocity.shape[-1] == x.shape[-1]

                    dx, dvec, _ = self.transformer_3d_layers[i](x_3d, velocity, edge_index=edge_index_3d, edge_feature=edge_feature_3d, edge_direction=edge_direction_3d, cutoff=cutoff_3d, node_mask=node_mask, self_attn_padding_mask=(node_features[:,:,0]==0), self_attn_mask=attn_mask)

                    x_3d = x_3d + dx
                    velocity = velocity + dvec
                    x = self.combine_proj(torch.cat([x, x_3d], dim=-1))
                    
                    x_3d = x
                    
                    
        ####################################################################################            
        if self.combine_transformer_output in {'cat_last'}:
            
            x_3d = x
            for i in range(0, self.num_layers):
                x, _, encoded_adj, graph_feature = self.transformer_2d_layers[i](x, adj=encoded_adj, graph_feature=graph_feature, node_mask=node_mask, edge_mask=edge_mask, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)


                if (self.transformer_3d_layers is not None) and (velocity is not None):

                    assert velocity.shape[-1] == x.shape[-1]

                    dx, dvec, _ = self.transformer_3d_layers[i](x_3d, velocity, edge_index=edge_index_3d, edge_feature=edge_feature_3d, edge_direction=edge_direction_3d, cutoff=cutoff_3d, node_mask=node_mask, self_attn_padding_mask=(node_features[:,:,0]==0), self_attn_mask=attn_mask)

                    x_3d = x_3d + dx
                    velocity = velocity + dvec
                    
                
            if (self.transformer_3d_layers is not None) and (velocity is not None):
                x = self.combine_proj(torch.cat([x, x_3d], dim=-1))
                    
                
                    
        x_out = x
        pos_out = None
        if (self.equivariant_output is not None) and (not (pos == 0).all()) and (velocity is not None):
            x_out, v_out = self.equivariant_output(x, velocity)
            pos_out = v_out.squeeze() + pos
            
        else:
            
            if (self.coordinate_proj is not None) and (not (pos == 0).all()) and (velocity is not None):
                pos_out = self.coordinate_proj(velocity).squeeze() + pos
            
            if self.out_node_proj is not None:
                x_out = self.out_node_proj(x)
            
        x_out = x_out * node_mask
        
        if pos_out is not None:
            pos_out = pos_out * node_mask
            
            
        adj_out = None
        if (self.out_edge_proj) is not None and (encoded_adj is not None):
            adj_out = self.out_edge_proj(encoded_adj)
            
            diag_mask = torch.eye(bs)
            diag_mask = ~diag_mask.type_as(adj).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
            adj_out = adj_out * edge_mask
            adj_out = 1/2 * (adj_out + torch.transpose(adj_out, 1, 2))
            
            
        
        return x_out, adj_out, pos_out