import torch
import torch.nn as nn
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from equivariant_diffusion import utils as diffusion_utils

class Transformer_dynamics(nn.Module):
    def __init__(self, model, context_node_dim, num_spatial=9, n_dims=3, condition_time=False):
        super().__init__()


        self.model = model

        self.context_node_dim = context_node_dim
        self.n_dims = n_dims
        self.condition_time = condition_time
        self.num_spatial = num_spatial

        
        
        
    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, edges, attn_bias, node_mask, edge_mask, context, use3d=True):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims if use3d else dims
        
        adj = edges[:, :, :, 1:].sum(-1).long()
        spatial_pos = diffusion_utils.get_spatial_positions(adj, edge_mask, self.num_spatial, xh.device)
        
        
        
        xh = xh.view(bs, n_nodes, -1).clone() * node_mask
        x = xh[:, :, 0:self.n_dims].clone() if use3d else None
            
    
        if h_dims == 0:
            h = torch.ones(bs, n_nodes, 1).to(xh.device)
        else:
            h = xh[:, :, self.n_dims:].clone() if use3d else xh.clone()

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
            context = context.view(bs, n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

            
        h_final, adj_final, x_final = self.model(node_features=h, attn_bias=attn_bias, spatial_pos=spatial_pos, edge_input=None, edge_type=edges, pos=x, adj=adj, node_mask=node_mask.view(bs, n_nodes, 1), edge_mask=edge_mask.view(bs, n_nodes, n_nodes, 1))
        
        vel = (x_final - x.view(bs, n_nodes, -1)) * node_mask.view(bs, n_nodes, 1)  # This masking operation is redundant but just in case
        

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if vel is not None:
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
                return torch.cat([vel, h_final], dim=2), adj_final
            
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return h_final, adj_final
