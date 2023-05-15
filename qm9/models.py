import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models import Transformer_dynamics

from equivariant_diffusion.en_diffusion import MuDiffusion
from egnn.ultimate_transformer_diffusion import TransformerEncoder

def get_prop_dist(args, dataloader_train):
    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    return prop_dist

def get_model(args, dataset_info):
    histogram = dataset_info['n_nodes']    
    max_nodes = dataset_info['max_n_nodes']
    max_weights = dataset_info['max_weight']
    edge_types = dataset_info['edge_types']
    atom_weights = dataset_info['atom_weights']
    max_in_deg = dataset_info['max_in_deg']
    max_out_deg = dataset_info['max_out_deg']
    max_num_edges = dataset_info['max_num_edges']
    max_spatial = dataset_info['max_spatial']
    max_edge_dist = dataset_info['max_edge_dist']

    
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)


    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf


    in_edge_nf = len(edge_types)

    transformer = TransformerEncoder(num_in_degree = max_in_deg,
                                    num_out_degree = max_out_deg,
                                    num_edges = max_num_edges,
                                    num_spatial = max_spatial,
                                    num_edge_dis = max_edge_dist,
                                    in_node_dim = dynamics_in_node_nf,
                                    in_edge_dim = in_edge_nf,
                                    max_weight = max_weights,
                                    atom_weights = atom_weights,
                                    edge_type = args.use_edge_type,
                                    multi_hop_max_dist = args.multi_hop_max_dist,
                                    num_encoder_layers = args.num_encoder_layers,
                                    embedding_dim = args.embedding_dim,
                                    edge_embedding_dim = args.edge_embedding_dim,
                                    num_attention_heads = args.num_attention_heads,
                                    use_3d_embedding = args.use_3d_embedding,
                                    num_3d_bias_kernel = args.num_3d_bias_kernel,
                                    use_2d_embedding = args.use_2d_embedding,
                                    use_2d_neighbor_embedding = args.use_2d_neighbor_embedding,
                                    use_3d_neighbor_embedding=args.use_3d_neighbor_embedding,
                                    apply_concrete_adjacency_neighbor = args.apply_concrete_adjacency_neighbor,
                                    use_2d_edge_embedding=args.use_2d_edge_embedding,
                                    cutoff_upper = args.cutoff_upper,
                                    cutoff_lower = args.cutoff_lower,
                                    distance_projection = args.distance_projection,
                                    trainable_dist_proj = args.trainable_dist_proj,
                                    neighbor_combine_embedding = args.neighbor_combine_embedding,
                                    use_extra_graph_embedding = args.use_extra_graph_embedding,
                                    use_extra_graph_embedding_attn_bias = args.use_extra_graph_embedding_attn_bias,
                                    extra_feature_type = args.extra_feature_type,
                                    graph_embedding_dim = args.graph_embedding_dim,
                                    ### transformer
                                    before_attention_dropout = args.before_attention_dropout,
                                    before_attention_layernorm = args.before_attention_layernorm,
                                    before_attention_quant_noise = args.before_attention_quant_noise,
                                    before_attention_qn_block_size = args.before_attention_qn_block_size,
                                    ffn_embedding_dim = args.ffn_embedding_dim,
                                    ffn_edge_embedding_dim = args.ffn_edge_embedding_dim,
                                    ffn_graph_embedding_dim = args.ffn_graph_embedding_dim,
                                    in_attention_feature_dropout = args.in_attention_feature_dropout,
                                    in_attention_dropout = args.in_attention_dropout,
                                    in_attention_activation_dropout = args.in_attention_activation_dropout,
                                    in_attention_activation_dropout_adj = args.in_attention_activation_dropout_adj,
                                    in_attention_activation_dropout_graph_feature=args.in_attention_activation_dropout_graph_feature,
                                    in_attention_activation_fn = nn.SiLU(),
                                    in_attention_quant_noise = args.in_attention_quant_noise,
                                    in_attention_qn_block_size = args.in_attention_qn_block_size,
                                    in_attention_layernorm = args.in_attention_layernorm,
                                    in_attention_droppath = args.in_attention_droppath,
                                    in_attention_droppath_adj = args.in_attention_droppath_adj,
                                    in_attention_droppath_graph_feature = args.in_attention_droppath_graph_feature,
                                    in_attention_pred_adjacency=args.in_attention_pred_adjacency,
                                    attention_activation_fn=args.attention_activation_fn,
                                    ### equivariant transformer
                                    use_equivariant_transformer = args.use_equivariant_transformer,
                                    equivariant_distance_influence = args.equivariant_distance_influence,
                                    equivariant_distance_activation_fn = nn.SiLU(),
                                    equivariant_attention_activation_fn = args.equivariant_attention_activation_fn,
                                    equivariant_in_attention_dropout = args.equivariant_in_attention_dropout,
                                    equivariant_use_x_layernorm = args.equivariant_use_x_layernorm,
                                    equivariant_use_dx_layernorm = args.equivariant_use_dx_layernorm,
                                    equivariant_dx_dropout = args.equivariant_dx_dropout,
                                    equivariant_coord_activation_fn = nn.SiLU(),
                                    equivariant_apply_concrete_adjacency = args.equivariant_apply_concrete_adjacency,
                                    ### output
                                    combine_transformer_output = args.combine_transformer_output,
                                    combine_transformer_activation_fn = nn.SiLU(),
                                    use_output_projection = args.use_output_projection,
                                    output_activation_fn = nn.SiLU(),
                                    out_node_dim = dynamics_in_node_nf,
                                    out_edge_dim = in_edge_nf,
                                    use_equivariant_output_projection = args.use_equivariant_output_projection,
                                    equivariant_output_activation_fn = nn.SiLU,
                                     )

    net_dynamics = Transformer_dynamics(
                        model=transformer, context_node_dim=args.context_node_nf, num_spatial=max_spatial,
                        n_dims=3, condition_time=True
                                        )
    
    
    if args.probabilistic_model == 'diffusion':
        vdm = MuDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges,
            edge_types=edge_types,
            num_edge_type = in_edge_nf,
            transition_type = 'marginal',
            )

        return vdm, nodes_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=args.weight_decay)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
