import torch
import torch.nn as nn
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np

from egnn.transformer import GraphTransformer
from egnn.egnn import EGNN
from torch_geometric.utils.sparse import dense_to_sparse

class Transformer_dynamics(nn.Module):
    def __init__(self, model, context_node_dim, n_dims=3, condition_time=False, device='cpu'):
        super().__init__()
        
        self.model = model

        self.context_node_dim = context_node_dim
        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time
        
        self.transformer = GraphTransformer()
        self.egnn = EGNN(6, 1, 64)


    def forward(self, t, xh, attn_bias, spatial_pos=None, edge_input=None, edge_type=None, adj=None, node_mask=None, edge_mask=None, context=None):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        xh = xh.view(bs, n_nodes, -1).clone() * node_mask
        x = xh[:, :, 0:self.n_dims].clone()
    
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, :, self.n_dims:].clone()


        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, :, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs, n_nodes, 1)
                
            h = torch.cat([h, h_time], dim=-1)
            
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)


        _, adj_final, _ = self.model(h, attn_bias, spatial_pos=spatial_pos, edge_input=edge_input, edge_type=edge_type, pos=x, adj=adj, node_mask=node_mask, edge_mask=edge_mask)
        
        h_final, x_final = self.egnn(h.view(bs*n_nodes, -1), x.view(bs*n_nodes, -1), dense_to_sparse(adj)[0], node_mask=node_mask.view(bs*n_nodes, 1), edge_mask=edge_mask.view(bs*n_nodes*n_nodes, 1))
        h_final = h_final.view(bs, n_nodes, -1)
        x_final = x_final.view(bs, n_nodes, -1)
        
        #h_final, adj_final, x_final = self.transformer(h, attn_bias, spatial_pos=spatial_pos, edge_input=edge_input, edge_type=edge_type, pos=x, adj=adj, node_mask=node_mask, edge_mask=edge_mask)
        #h_final, adj_final = self.transformer(h, edge_type, torch.rand(bs, 0), node_mask, edge_mask)
        #x_final = x
        
        
        
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        adj_final = adj_final * edge_mask


        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :, :-1]


        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))


        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=-1), adj_final
