from egnn.ultimate_transformer import TransformerMEncoder
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
import algos
import numpy as np
import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F



bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
mol = Chem.MolFromSmiles('CCN1CCN(Cc2ccc(Nc3ncc(F)c(-c4cc(F)c5nc(C)n(C(C)C)c5c4)n3)nc2)CC1')
row, col, edge_type = [], [], []
for bond in mol.GetBonds():
    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    row += [start, end]
    col += [end, start]
    edge_type += 2 * [bonds[bond.GetBondType()] + 1]

edge_index = torch.tensor([row, col], dtype=torch.long)
edge_type = torch.tensor(edge_type, dtype=torch.long)
edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).long()
adj = torch_geometric.utils.to_dense_adj(edge_index)[0].long()

n_nodes = adj.size(1)
attn_edge_type = torch.zeros([n_nodes, n_nodes, edge_attr.size(-1)], dtype=torch.long)
attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr
shortest_path_result, path = algos.floyd_warshall(adj.numpy())
max_dist = np.amax(shortest_path_result)
spatial_pos = torch.from_numpy((shortest_path_result)).long()
edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())+1
edge_input = torch.from_numpy(edge_input).long()
attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

node_atom = torch.ones([1, n_nodes, 5])
node_charge = torch.ones([1, n_nodes, 1])
pos = torch.rand([1, n_nodes, 3])



transformer = TransformerMEncoder(num_atoms = 5,
                                    num_charges = 8,
                                    num_in_degree = 40,
                                    num_out_degree = 40,
                                    num_edges = 300,
                                    num_spatial = 40,
                                    num_edge_dis = 40,
                                    max_weight=200,
                                    atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19},
                                    edge_type = 'multi_hop',
                                    multi_hop_max_dist = 2,
                                    num_encoder_layers = 3,
                                    embedding_dim = 32,
                                    edge_embedding_dim = 16,
                                    num_attention_heads = 4,
                                    use_3d_embedding = True,
                                    num_3d_bias_kernel = 16,
                                    use_2d_embedding = True,
                                    use_2d_neighbor_embedding=True,
                                    use_3d_neighbor_embedding=True,
                                    apply_concrete_adjacency_neighbor = True,
                                    use_2d_edge_embedding=True,
                                    cutoff_upper = 5.0,
                                    cutoff_lower = 0.0,
                                    max_num_neighbors = 20,
                                    distance_projection = 'exp',
                                    trainable_dist_proj = True,
                                    neighbor_combine_embedding='add',
                                    use_extra_graph_embedding = True,
                                    use_extra_graph_embedding_attn_bias = True,
                                    extra_feature_type = 'all',
                                    graph_embedding_dim = 8,
                                    ### transformer
                                    before_attention_dropout=0.1,
                                    before_attention_layernorm=True,
                                    before_attention_quant_noise= 0,
                                    before_attention_qn_block_size= 0,
                                    ffn_embedding_dim = 80,
                                    ffn_edge_embedding_dim = 64,
                                    ffn_graph_embedding_dim = 24,
                                    in_attention_feature_dropout=0.1,
                                    in_attention_dropout=0.1,
                                    in_attention_activation_dropout=0.1,
                                    in_attention_activation_dropout_adj=0.1,
                                    in_attention_activation_dropout_graph_feature=0.1,
                                    in_attention_activation_fn=nn.SiLU(),
                                    in_attention_quant_noise = 0,
                                    in_attention_qn_block_size = 0,
                                    in_attention_layernorm = True,
                                    in_attention_droppath = 0.1,
                                    in_attention_droppath_adj = 0.1,
                                    in_attention_droppath_graph_feature = 0.1,
                                    in_attention_pred_adjacency=True,
                                    attention_activation_fn='softmax',
                                    ### equivariant transformer
                                    use_equivariant_transformer=True,
                                    num_equivariant_encoder_layers=2,
                                    equivariant_distance_influence='both',
                                    equivariant_distance_activation_fn=nn.SiLU(),
                                    equivariant_attention_activation_fn = 'softmax',
                                    equivariant_use_x_layernorm = True,
                                    equivariant_use_dx_layernorm = False,
                                    equivariant_dx_dropout = 0.0,
                                    equivariant_coord_activation_fn = nn.SiLU(),
                                    equivariant_apply_concrete_adjacency = True,
                                    ### output
                                    combine_transformer_output = 'cat',
                                    combine_transformer_activation_fn = nn.SiLU(),
                                    use_output_projection = True,
                                    output_activation_fn = nn.SiLU(),
                                    out_node_dim = 32,
                                    out_edge_dim = 32,
                                 )

print('num parameters: ', sum(p.numel() for p in transformer.parameters()))

node_mask = torch.ones([1, n_nodes, 1])
edge_mask = torch.ones([1, n_nodes, n_nodes, 1])

transformer(node_atom.repeat(2, 1, 1), node_charge.repeat(2, 1, 1).long(), attn_bias.unsqueeze(0).repeat(2, 1, 1), spatial_pos.unsqueeze(0).repeat(2, 1, 1), edge_input.unsqueeze(0).repeat(2, 1, 1, 1, 1), attn_edge_type.unsqueeze(0).repeat(2, 1, 1, 1), torch.cat([torch.rand([1, n_nodes, 3]), torch.rand([1, n_nodes, 3])+1], dim=0), adj.unsqueeze(0).repeat(2, 1, 1), node_mask.repeat(2, 1, 1), edge_mask.repeat(2, 1, 1, 1))