import torch

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
import algos
 
import numpy as np

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if (len(props.shape) == 3) and (props.shape[1] == props.shape[2]):
        return props[:, to_keep, :][:, :, to_keep]
    
    elif (len(props.shape) > 3) and (props.shape[1] == props.shape[2]):
        return props[:, to_keep, :, ...][:, :, to_keep, ...]
        
    else:
        if not torch.is_tensor(props[0]):
            return props
        elif props[0].dim() == 0:
            return props
        else:
            return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        to_keep = (batch['charges'].sum(0) > 0)
        


        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
        

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size, n_nodes, n_nodes, 1)
        
        
        spatial_positions = []
        #edge_inputs = []
        # compute edge inputs
        for (adj, edge_type, mask) in zip(batch['adj'], batch['edge_attr'], batch['edge_mask']):
            mask = mask.squeeze().numpy()
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            shortest_path_result = shortest_path_result * mask
            path = path * mask
            max_dist = np.amax(shortest_path_result)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            
            #edge_input = algos.gen_edge_input(max_dist, path, edge_type[:, :, 1:].numpy())+1
            #edge_input = torch.from_numpy(edge_input).long()
            
            spatial_positions.append(spatial_pos)
            #edge_inputs.append(edge_input)
            


        spatial_positions = torch.stack(spatial_positions) * edge_mask
        
        #max_val = max([t.shape[2] for t in edge_inputs])
        #edge_inputs = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_val - t.shape[2])) for t in edge_inputs])
        #edge_inputs = torch.cat([torch.zeros(batch_size, n_nodes, n_nodes, max_val, 1), edge_inputs], dim=-1)
        
        #edge_inputs = edge_inputs * edge_mask.unsqueeze(-1).unsqueeze(-1)    
        
        batch['spatial_positions'] = spatial_positions
        #batch['edge_input'] = edge_inputs
        batch['attn_bias'] = torch.zeros([batch_size, n_nodes + 1, n_nodes + 1], dtype=torch.float)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch
