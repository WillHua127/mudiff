import torch
from torch import nn

class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        x_out_channels,
        v_out_channels,
        intermediate_channels=None,
        activation=nn.SiLU(),
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.x_out_channels = x_out_channels
        self.hidden = hidden_channels
        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, v_out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            activation,
            nn.Linear(intermediate_channels, x_out_channels+v_out_channels),
        )

        self.act = activation if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        bs, n_nodes = x.shape[:2]
        vec1_buffer = self.vec1_proj(v).view(bs*n_nodes, 3, -1)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            bs*n_nodes, self.hidden, device=vec1_buffer.device
        )
        mask = (vec1_buffer != 0).view(bs*n_nodes, -1).any(dim=1)        
        if not mask.all():
            warnings.warn(
                (
                    f"Skipping gradients for {(~mask).sum()} atoms due to vector features being zero. "
                    "This is likely due to atoms being outside the cutoff radius of any other atom. "
                    "These atoms will not interact with any other atom unless you change the cutoff."
                )
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)
        vec1 = vec1.view(bs, n_nodes, -1)

        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.x_out_channels, dim=-1)
        v = v.unsqueeze(2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
    
    
    
# class OutputModel(nn.Module, metaclass=ABCMeta):
#     def __init__(self, allow_prior_model, reduce_op):
#         super(OutputModel, self).__init__()
#         self.allow_prior_model = allow_prior_model
#         self.reduce_op = reduce_op

#     def reset_parameters(self):
#         pass

#     @abstractmethod
#     def pre_reduce(self, x, v, z, pos, batch):
#         return

#     def reduce(self, x, batch):
#         return scatter(x, batch, dim=0, reduce=self.reduce_op)

#     def post_reduce(self, x):
#         return x
    
    
    
class EquivariantScalar(nn.Module):
    def __init__(
        self,
        hidden_channels,
        x_out,
        v_out,
        activation=nn.SiLU(),
    ):
        super(EquivariantScalar, self).__init__( )
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, x_out, v_out, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        print(x.shape, v.shape)
        return x, v