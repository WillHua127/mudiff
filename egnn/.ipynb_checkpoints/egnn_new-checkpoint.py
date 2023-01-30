from torch import nn
import torch
import math
from egnn.mlp import MLP
from egnn.gnn import GIN
from egnn.transformer import GraphTransformer
#from equivariant_diffusion.utils import mask_adjs, node_feature_to_matrix

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', adj_type='digress'):
        super(EGNN, self).__init__()
        self.device = device
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.in_node_nf = in_node_nf

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        
        
        
        self.adj_type = adj_type
        if self.adj_type == 'dppm':
            self.embedding2d = nn.Linear(in_node_nf, hidden_nf)
            self.embedding_out2d = nn.Linear(hidden_nf, out_node_nf)
            self.channel_num = 2
            

            gin0 = GIN([hidden_nf+self.channel_num*2], dropout_p=0.0, out_dim=hidden_nf, use_norm_layers=False, channel_num=self.channel_num)
            self.edge_models = EdgeDensePredictionGNNLayer(gin0, c_in = self.channel_num, c_out = self.channel_num, num_classes=None)
            
            self.final_read_score = MLP(input_dim=2*2, output_dim=1, activate_func=nn.functional.elu,
                                    hidden_dim=4, num_layers=3, num_classes=None)
            
            
        elif self.adj_type == 'digress':            
            self.transformer = GraphTransformer(n_layers=2, in_node_nf=5, in_edge_nf=2, in_y_nf=11,
                                                hidden_node_mlp=256, hidden_edge_mlp=128, hidden_y_mlp=128,
                                                hidden_node_nf=256, hidden_edge_nf=64, hidden_y_nf=64, n_head=8, ff_node_nf=256, ff_edge_nf=128,
                                                out_node_nf=1, out_edge_nf=2, out_y_nf=0, act_fn=act_fn)

        
        self.equivariant = EGNNetwork(in_node_nf, hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                      act_fn=act_fn, n_layers=inv_sublayers,
                                      attention=attention, norm_diff=norm_diff, tanh=tanh,
                                      coords_range=coords_range, norm_constant=norm_constant,
                                      sin_embedding=self.sin_embedding,
                                      normalization_factor=self.normalization_factor,
                                      aggregation_method=self.aggregation_method)
        

        self.to(self.device)

    def forward(self, h, x, edge_index, bs, n_nodes, h_2d=None, y_2d=None, adjs=None, node_mask=None, edge_mask=None, compute_adj=True):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        
        if self.adj_type == 'dppm':
            if compute_adj:
                ori_adjs = edge_mask.view(bs, n_nodes, n_nodes).unsqueeze(1)
                adjs = torch.cat([ori_adjs, 1. - ori_adjs], dim=1)  # B x 2 x N x N
                adjs = mask_adjs(adjs, node_mask.view(bs, n_nodes))
                temp_adjs = [adjs]
                h_2d = self.embedding2d(h.view(bs, n_nodes, self.in_node_nf))
                h_2d, adjs = self.edge_models(h_2d, adjs, node_mask.view(bs, n_nodes))
                temp_adjs.append(adjs)
                
        elif self.adj_type == 'digress':
            h_2d, pred_adjs, y_2d = self.transformer(h_2d, adjs, y_2d, node_mask, edge_mask)
            
            if node_mask is not None:
                h_2d = h_2d * node_mask.view(bs, n_nodes, 1)
                pred_adjs = pred_adjs * edge_mask.view(bs, n_nodes, n_nodes, 1)

            
        h_3d, x = self.equivariant(h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
            
        if node_mask is not None:
            h_3d = h_3d * node_mask
            
            
        if self.adj_type == 'dppm':
            if compute_adj:
                stacked_adjs = torch.cat(temp_adjs, dim=1)
                mlp_in = stacked_adjs.permute(0, 2, 3, 1)
                out_shape = mlp_in.shape[:-1]
                mlp_out = self.final_read_score(mlp_in)
                score = mlp_out.view(*out_shape)
                return h_3d, x, score
            
        elif self.adj_type == 'digress':
            return h_3d, x, pred_adjs
        
        return h_3d, x, None

    
class EGNNetwork(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
            
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        self.embedding_in = nn.Sequential(nn.Linear(in_node_nf, hidden_nf), act_fn,
                                      nn.Linear(hidden_nf, hidden_nf), act_fn)
        
        self.embedding_out = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), act_fn,
                                       nn.Linear(hidden_nf, out_node_nf))
        
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
            
        
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr)
            
        h = self.embedding_out(h)
        return h, x
    
    
class EdgeDensePredictionGNNLayer(nn.Module):
    def __init__(self, gnn_module, c_in, c_out,
                 num_classes=1):
        super().__init__()
        self.multi_channel_gnn_module = gnn_module
        self.translate_mlp = MLP(num_layers=3, input_dim=c_in + 2 * gnn_module.get_out_dim(),
                                 hidden_dim=max(c_in, c_out) * 2, output_dim=c_out,
                                 activate_func=nn.functional.elu,
                                 use_bn=True,
                                 num_classes=num_classes)

    def forward(self, x, adjs, node_flags):
        x_o = self.multi_channel_gnn_module(x, adjs, node_flags)  # B x N x F_o
        x_o_pair = node_feature_to_matrix(x_o)  # B x N x N x 2F_o
        last_c_adjs = adjs.permute(0, 2, 3, 1)  # B x N x N x C_i
        mlp_in = torch.cat([last_c_adjs, x_o_pair], dim=-1)  # B x N x N x (2F_o+C_i)
        mlp_in_shape = mlp_in.shape
        mlp_out = self.translate_mlp(mlp_in.view(-1, mlp_in_shape[-1]))
        new_adjs = mlp_out.view(mlp_in_shape[0], mlp_in_shape[1], mlp_in_shape[2], -1).permute(0, 3, 1, 2)
        new_adjs = new_adjs + new_adjs.transpose(-1, -2)
        # new_adjs = torch.sigmoid(new_adjs)
        new_adjs = mask_adjs(new_adjs, node_flags)
        return x_o, new_adjs
    

    
class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
