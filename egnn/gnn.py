import torch
import torch.nn as nn
import torch.nn.functional as F

from egnn.mlp import MLP


def add_self_loop_if_not_exists(adjs):
    if len(adjs.shape) == 4:
        return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).unsqueeze(0).to(adjs.device)
    return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)

def check_adjs_symmetry(adjs):
    if not do_check_adjs_symmetry:
        return
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum([0, 1, 2]) < 1e-2

    
    
class GraphNeuralNetwork(nn.Module):

    def _aggregate(self, x, adjs, node_flags, layer_k):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :param layer_k: an int, the index of the layer
        :return: a: B x N x F_mid, the aggregated feature vectors of the neighbors of node
        """
        return x

    def _combine(self, x, a, layer_k):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :return: a: B x N x F_mid, the aggregated feature vectors of the neighbors of node
        :return: x: B x N x F_out, the feature vectors of nodes
        """
        return a

    @staticmethod
    def _graph_preprocess(x, adjs, node_flags):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :return:
            x: B x N x F_in, the feature vectors of nodes
            adjs: B x N x N, the adjacent matrix, with self-loop
            node_flags: B x N, the flags for the existence of nodes
        """
        x = (x * node_flags.unsqueeze(-1))
        check_adjs_symmetry(adjs)
        return x, adjs, node_flags

    @staticmethod
    def _readout(x, adjs, node_flags):
        """

        :param x: B x N x F_in, the feature vectors of nodes
        :param adjs: B x N x N, the adjacent matrix, with self-loop
        :param node_flags: B x N, the flags for the existence of nodes
        :return: energy: B, a float number as the energy of each graph
        """
        x = (x * node_flags.unsqueeze(-1))
        return x.view(x.size(0), -1).sum(-1).squeeze()

    def __init__(self, max_layers_num, channel_num):
        super().__init__()
        self.max_layers_num = max_layers_num
        self.deg_projection = nn.Linear(channel_num, 2*channel_num)


    def get_node_feature(self, x, adjs, node_flags):
        deg = adjs.sum(-1).unsqueeze(-1)  # B x C x N x 1 or B x N x 1
        if len(deg.shape) == 4:
            deg = deg.permute(0, 2, 1, 3).contiguous().view(adjs.size(0), adjs.size(-1), -1)
        
        deg = self.deg_projection(deg)
        if x is None:
            x = deg
        else:
            x = torch.cat([x, deg], dim=-1)
        x, adjs, node_flags = self._graph_preprocess(x, adjs, node_flags)
        for k in range(self.max_layers_num):
            x = self._combine(x=x, a=self._aggregate(x=x, adjs=adjs, node_flags=node_flags, layer_k=k), layer_k=k)
        return x

    def forward(self, x, adjs, node_flags):
        x = self.get_node_feature(x, adjs, node_flags)
        return self._readout(x, adjs, node_flags)  # energy for each graph
    
    


    
class GIN(GraphNeuralNetwork):

    def __init__(self, feature_nums, dropout_p=0.5, out_dim=1, use_norm_layers=True, channel_num=1):
        self._out_dim = out_dim
        self.channel_num = channel_num
        self.feature_nums = feature_nums
        hidden_num = 2 * max(feature_nums)

        def linear_with_leaky_relu(ii):
            return nn.Sequential(nn.Linear(feature_nums[ii], hidden_num),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_num, out_dim))

        layer_n = len(feature_nums) - 1
        super().__init__(max_layers_num=layer_n, channel_num=channel_num)
        self.use_norm_layers = use_norm_layers
        self.eps = nn.Parameter(torch.zeros(layer_n))
        if self.use_norm_layers:
            self.norm_layers = torch.nn.ModuleList()
        self.linear_prediction = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        

        for i in range(layer_n):
            mlp = MLP(num_layers=2, input_dim=feature_nums[i] * channel_num,
                          hidden_dim=hidden_num,
                          output_dim=feature_nums[i + 1])
            self.layers.append(mlp)
            if self.use_norm_layers:
                self.norm_layers.append(nn.BatchNorm1d(feature_nums[i + 1]))
                # self.norm_layers.append(
                #     ConditionalNorm1dPlus(num_features=feature_nums[i],
                #                                   num_classes=1))
            self.linear_prediction.append(linear_with_leaky_relu(i))
        self.linear_prediction.append(linear_with_leaky_relu(-1))

        self.dropout_p = dropout_p
        self.hidden = []

    def get_out_dim(self):
        return self._out_dim

    def _aggregate(self, x, adjs, node_flags, layer_k):
        batch_size = x.size(0)
        feature_num = x.size(-1)
        if self.use_norm_layers:
            x = self.norm_layers[layer_k](x.view(-1, feature_num)).contiguous().view(batch_size, -1, feature_num)

        if len(adjs.shape) == 4:
            h = torch.matmul(adjs, x.unsqueeze(1))  # B x C x N x F
            h = h.permute(0, 2, 1, 3).contiguous().view(adjs.size(0), adjs.size(-1), -1)  # B x N x CF
        else:
            h = torch.bmm(adjs, x)
        h = h + self.eps[layer_k] * torch.cat([x]*self.channel_num, dim=-1)

        feature_num = h.size(-1)
        h = h.view(-1, feature_num)
        # print(h.size())
        # print(self.layers[layer_k])
        h = self.layers[layer_k](h)
        h = torch.tanh(h)

        h = h.view(batch_size, -1, h.size(-1))


        self.hidden.append((h * node_flags.unsqueeze(-1)))

        return h

    def _combine(self, x, a, layer_k):
        return a

    def _readout(self, x, adjs, node_flags):
        ret = 0.
        for layer, h in enumerate(self.hidden):
            ret = ret + F.dropout(
                self.linear_prediction[layer](h),
                self.dropout_p,
                training=self.training
            )
        return ret.squeeze(-1)  # B x N x F_out

    def _graph_preprocess(self, x, adjs, node_flags):
        adjs = add_self_loop_if_not_exists(adjs)
        # d = adjs.sum(dim=-1)
        # d -= d.min(dim=-1, keepdim=True).values
        # d += 1e-5
        # dh = torch.sqrt(d).reciprocal()
        # adj_hat = dh.unsqueeze(1) * adjs * dh.unsqueeze(-1)
        # adj_hat = torch.softmax(adjs, dim=-1)
        # adj_hat = doubly_stochastic_norm(adjs, do_row_norm=True)
        adj_hat = adjs
        x = (x * node_flags.unsqueeze(-1))

        self.hidden = []
        self.hidden.append(x)
        return x, adj_hat, node_flags

    def get_node_feature(self, x, adjs, node_flags):
        super().get_node_feature(x, adjs, node_flags)
        node_features = torch.cat(self.hidden, dim=-1)
        return node_features