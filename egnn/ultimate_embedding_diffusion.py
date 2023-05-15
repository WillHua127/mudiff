import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric

import math

from egnn.ultimate_spatial_feature import *


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
            
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
       
    
@torch.jit.script
def gaussian(x, mean, std):
    pi = math.pi
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512*3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul.unsqueeze(-1) * x.unsqueeze(-1) + bias.unsqueeze(-1)
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

    
class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
    
    
class ExponentialLayer(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExponentialLayer, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
    
    
class CombineEmbedding(nn.Module):

    def __init__(self, hidden_dim, n_layers, neighbor_combine_embedding='cat', use_2d=False, use_3d=False):
        super(CombineEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_combine_embedding = neighbor_combine_embedding
        
        self.scale = 1
        if use_2d:
            self.scale += 1
            
        if use_3d:
            self.scale += 1
            
        if self.neighbor_combine_embedding == 'no':
            self.scale = 1

        self.combine = None
        if self.neighbor_combine_embedding == 'cat':
            if self.scale > 1:
                self.combine = nn.Linear(hidden_dim * self.scale, hidden_dim)
            

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, x_2d_neighbor=None, x_3d_neighbor=None):   
        
        if self.scale == 1:
            return x
        
        
        if (x_2d_neighbor is not None) and (x_3d_neighbor is not None):
            if self.neighbor_combine_embedding == 'cat':
                x[:, 1:, :] = self.combine(torch.cat([x[:, 1:, :], x_2d_neighbor, x_3d_neighbor], dim=-1))
                
            elif self.neighbor_combine_embedding == 'add':
                x[:, 1:, :] = x[:, 1:, :] + x_2d_neighbor + x_3d_neighbor
                
            return x
        
        
        if (x_2d_neighbor is None) and (x_3d_neighbor is not None):
            if self.neighbor_combine_embedding == 'cat':
                x[:, 1:, :] = self.combine(torch.cat([x[:, 1:, :], x_3d_neighbor], dim=-1))
                
            elif self.neighbor_combine_embedding == 'add':
                x[:, 1:, :] = x[:, 1:, :] + x_3d_neighbor
                    
            return x
            
            
        if (x_2d_neighbor is not None) and (x_3d_neighbor is None):
            if self.neighbor_combine_embedding == 'cat':
                padding = (0, self.hidden_dim)
                x[:, 1:, :] = self.combine(torch.cat([F.pad(x[:, 1:, :], padding, value=0).to(x.device), x_2d_neighbor], dim=-1))
                
            elif self.neighbor_combine_embedding == 'add':
                x[:, 1:, :] = x[:, 1:, :] + x_2d_neighbor
                
            return x

        return x
    
    
class AtomEmbedding(nn.Module):

    def __init__(self, in_node_dim, num_in_degree, num_out_degree, hidden_dim, n_layers, use_2d=True):
        super(AtomEmbedding, self).__init__()
        self.in_node_dim = in_node_dim
        self.use_2d = use_2d

        # 1 for graph token
        self.atom_encoder = nn.Linear(in_node_dim, hidden_dim, bias=False)
        self.in_degree_encoder = nn.Embedding(num_in_degree + 1, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree + 1, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, adj=None):
        bs, n_nodes = node_features.size()[:2]
        
        # node feauture + graph token
        node_atom_feature = self.atom_encoder(node_features) # [n_graph, n_node, n_hidden]
        node_feature = node_atom_feature

        if self.use_2d and (adj is not None):
            in_degree = adj.sum(2).long()
            out_degree = adj.sum(1).long()
            degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            node_feature = node_feature + degree_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(bs, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature
    
    
    
class EdgeEmbedding(nn.Module):

    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, edge_hidden_dim, n_layers):
        super(EdgeEmbedding, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim

        # 1 for graph token
        self.atom_encoder = nn.Linear(in_node_dim, hidden_dim, bias=False)
        self.edge_encoder = nn.Linear(in_edge_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(hidden_dim*2, edge_hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, adj, attn_edge_type):
        bs, n_nodes = node_features.size()[:2]

        
        # node feauture
        node_atom_feature = self.atom_encoder(node_features) # [n_graph, n_node, n_hidden]
        node_feature = node_atom_feature
        
        node_feature = node_feature.unsqueeze(1) * node_feature.unsqueeze(2)
        edge_feature = self.edge_encoder(attn_edge_type)
        edge_feature = torch.cat([edge_feature, node_feature], dim=-1)
        edge_feature = self.edge_embedding(edge_feature)
        
        # ensure symmetry
        encoded_adj = (edge_feature + edge_feature.transpose(1, 2)) / 2 # symmetric [n_graph, n_node, n_node, n_hidden]
        
        return encoded_adj
    
    
    
class ExtraGraphFeatureEmebedding(nn.Module):

    def __init__(self, max_n_nodes, max_weight, atom_weights, extra_feature_type, graph_hidden_dim, n_layers, use_2d):
        super(ExtraGraphFeatureEmebedding, self).__init__()
        self.max_n_nodes = max_n_nodes
        self.extra_features_type = extra_feature_type
        self.use_extra_molecular_feature = True if atom_weights is not None else False
        self.num_atom_type = len(atom_weights) if atom_weights is not None else 0
        self.use_2d = use_2d
        
        self.extra_features = ExtraFeatures(extra_features_type=extra_feature_type, max_n_nodes=max_n_nodes)  if self.use_2d else None
        
        self.extra_molecular_features = ExtraMolecularFeatures(max_weight=max_weight, atom_weights=atom_weights) if self.use_extra_molecular_feature else None
        
        scale = 0
        if self.use_extra_molecular_feature:
            scale += 1
            
        if self.use_2d:
            if self.extra_features_type in {'all'}:
                scale += 11
                
            elif self.extra_features_type in {'cycles'}:
                scale += 5
                
            elif self.extra_features_type in {'eigenvalues'}:
                scale += 7
            
        self.graph_encoder = nn.Linear(scale, graph_hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, adj=None, edge_type=None, node_mask=None):
        
        spatial_graph_embedding = self.extra_features(edge_type, node_mask.squeeze(-1).bool()) if (self.extra_features is not None) and (adj is not None) else None
                
        molecular_graph_embedding = self.extra_molecular_features(node_features[:, :, :self.num_atom_type]) if (self.extra_molecular_features is not None) else None
        
        
        if (spatial_graph_embedding is not None) and (molecular_graph_embedding is not None):
            graph_embedding = torch.cat([spatial_graph_embedding, molecular_graph_embedding], dim=-1)
        
        elif (spatial_graph_embedding is None) and (molecular_graph_embedding is not None):
            graph_embedding = molecular_graph_embedding
            
        elif (spatial_graph_embedding is not None) and (molecular_graph_embedding is None):
            graph_embedding = spatial_graph_embedding
        
        else:
            return None
        
        graph_embedding = self.graph_encoder(graph_embedding)
        return graph_embedding
        
    
    
    
class MoleculeAttnBias(nn.Module):
    def __init__(self, num_heads, in_edge_dim, num_spatial, num_edge_dis, hidden_dim, edge_type, multi_hop_max_dist, n_layers, use_2d):
        super(MoleculeAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.use_2d = use_2d

        self.edge_encoder = nn.Linear(in_edge_dim, num_heads, bias=False)

        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial+1, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, attn_bias, spatial_pos, edge_input, attn_edge_type):

        n_graph, n_node = node_features.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if self.use_2d:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if self.use_2d:
            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)

            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(attn_edge_type).permute(0, 3, 1, 2)


            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
    
    
class MoleculeNeighbor2DEmbedding(torch_geometric.nn.MessagePassing):
    def __init__(self, in_node_dim, in_edge_dim, n_layers, hidden_dim):
        super(MoleculeNeighbor2DEmbedding, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim

        self.atom_encoder = nn.Linear(in_node_dim, hidden_dim, bias=False)        
        self.edge_encoder = nn.Linear(in_edge_dim, hidden_dim, bias=False) 

        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
    def forward(self, node_features, adj, edge_feature):
        
        bs, n_nodes = node_features.size()[:2]
        
        
        # node feauture + graph token
        node_atom_feature = self.atom_encoder(node_features) # [n_graph, n_node, n_hidden]
        node_feature = node_atom_feature
        
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_weight = self.edge_encoder(edge_feature) * adj.unsqueeze(-1)
        edge_weight = edge_weight[edge_weight != 0].view(-1, self.hidden_dim)

        x_neighbors = self.propagate(edge_index, x=node_feature.view(-1, self.hidden_dim), W=edge_weight, size=None)
        x_neighbors = x_neighbors.view(bs, n_nodes, self.hidden_dim)
                
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W
    
    

class Molecule3DBias(nn.Module):
    def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, cutoff_lower=0.0, cutoff_upper=5.0, distance_projection='exp', trainable=True, no_share_rpe=False):
        super(Molecule3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.distance_projection = distance_projection
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper


        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        
        if distance_projection == 'exp':
            self.dist_proj = ExponentialLayer(cutoff_lower, cutoff_upper, self.num_kernel, trainable)
            self.nonlin_proj = NonLinear(self.num_kernel, rpe_heads)
        elif distance_projection == 'gaussian':
            self.dist_proj = GaussianLayer(self.num_kernel, num_edges)
            self.nonlin_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None
            
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, node_features, pos):

        padding_mask = node_features.eq(0).all(dim=-1)
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        radial = torch.sum(delta_pos**2, -1)#.unsqueeze(-1)
        dist = torch.sqrt(radial)
        dist = torch.nan_to_num(dist)


        if self.distance_projection == 'exp':
            edge_feature = self.dist_proj(dist)
            
        elif self.distance_projection == 'gaussian':
            edge_feature = self.dist_proj(dist, torch.zeros_like(dist).long())
            
        nonlin_result = self.nonlin_proj(edge_feature)
        graph_attn_bias = nonlin_result
        
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float(-1e20)
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)        

        return graph_attn_bias, merge_edge_features, delta_pos
    
    
    
    
class MoleculeNeighbor3DEmbedding(torch_geometric.nn.MessagePassing):
    def __init__(self, in_node_dim, num_edges, n_layers, num_kernel, hidden_dim, cutoff_lower=0.0, cutoff_upper=5.0, distance_projection='exp', trainable=True, use_2d=False):
        super(MoleculeNeighbor3DEmbedding, self).__init__()
        self.in_node_dim = in_node_dim
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.hidden_dim = hidden_dim
        self.distance_projection = distance_projection
        self.use_2d = use_2d
        
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower

        self.atom_encoder = nn.Linear(in_node_dim, hidden_dim, bias=False)

        
        if distance_projection == 'exp':
            self.dist_proj = ExponentialLayer(cutoff_lower, cutoff_upper, self.num_kernel, trainable)
            self.nonlin_proj = NonLinear(self.num_kernel, hidden_dim)
        elif distance_projection == 'gaussian':
            self.dist_proj = GaussianLayer(self.num_kernel, num_edges)
            self.nonlin_proj = NonLinear(self.num_kernel, hidden_dim)
        
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
    def forward(self, node_features, pos, adj=None):
        
        bs, n_nodes = node_features.size()[:2]
                
        # node feauture + graph token
        node_atom_feature = self.atom_encoder(node_features) # [n_graph, n_node, n_hidden]
        node_feature = node_atom_feature
        

        # edge_vec = pos.unsqueeze(1) - pos.unsqueeze(2)
        # dist = edge_vec.norm(dim=-1).view(-1, n_nodes, n_nodes)
        # dist = torch.nan_to_num(dist)
        
        delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        radial = torch.sum(delta_pos**2, -1)
        dist = torch.sqrt(radial)
        
        cutoffs = 0.5 * (torch.cos(dist * math.pi / self.cutoff_upper) + 1.0)
        cutoffs = cutoffs * (dist < self.cutoff_upper).float() * (dist > self.cutoff_lower).float()
        
        if self.use_2d and (adj is not None):
            cutoffs = torch.where(((adj>0) + cutoffs) < 1, cutoffs, torch.tensor(1.).to(pos.device))
        
        edge_index, _ = torch_geometric.utils.dense_to_sparse(cutoffs)
        
        
        if self.distance_projection == 'exp':
            edge_feature = self.dist_proj(dist)
            
        elif self.distance_projection == 'gaussian':
            edge_feature = self.dist_proj(dist, torch.zeros_like(dist).long())
            
            
        edge_weight = self.nonlin_proj(edge_feature)
        edge_weight = edge_weight.view(-1, self.hidden_dim)
        edge_weight = edge_weight[cutoffs.view(-1) > 0]

        
        x_neighbors = self.propagate(edge_index, x=node_feature.view(-1, self.hidden_dim), W=edge_weight, size=None)
        x_neighbors = x_neighbors.view(bs, n_nodes, self.hidden_dim)
                
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W
        
        
        
        
class GeometricInformation(nn.Module):
    def __init__(self, num_kernel, num_edges, n_layers, cutoff_lower=0.0, cutoff_upper=5.0, distance_projection='exp', trainable=True, use_2d=False):
        super(GeometricInformation, self).__init__()
        self.num_kernel = num_kernel
        self.distance_projection = distance_projection
        self.use_2d = use_2d
        
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower
        
        if distance_projection == 'exp':
            self.dist_proj = ExponentialLayer(cutoff_lower, cutoff_upper, self.num_kernel, trainable)
        elif distance_projection == 'gaussian':
            self.dist_proj = GaussianLayer(self.num_kernel, num_edges)
        
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
    def forward(self, pos, adj=None):
        
        bs, n_nodes = pos.size()[:2]
            
        delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        radial = torch.sum(delta_pos**2, -1)
        dist = torch.sqrt(radial)
        
        cutoffs = 0.5 * (torch.cos(dist * math.pi / self.cutoff_upper) + 1.0)
        cutoffs = cutoffs * (dist < self.cutoff_upper).float() * (dist > self.cutoff_lower).float()
        
        if self.use_2d and (adj is not None):
            cutoffs = torch.where(((adj>0) + cutoffs) < 1, cutoffs, torch.tensor(1.).to(pos.device))
        
        edge_index, _ = torch_geometric.utils.dense_to_sparse(cutoffs)
        
        
        if self.distance_projection == 'exp':
            edge_feature = self.dist_proj(dist)
            
        elif self.distance_projection == 'gaussian':
            edge_feature = self.dist_proj(dist, torch.zeros_like(dist).long())
            
        
        norm = dist.detach() + 1e-10
        delta_pos = delta_pos/norm.unsqueeze(-1)
        
        return edge_index, edge_feature, delta_pos, cutoffs
        
