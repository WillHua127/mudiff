import logging
import os
import torch
import numpy as np
import tarfile
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import subgraph, to_dense_adj

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
import algos



def split_dataset(data, split_idxs):
    """
    Splits a dataset according to the indices given.

    Parameters
    ----------
    data : dict
        Dictionary to split.
    split_idxs :  dict
        Dictionary defining the split.  Keys are the name of the split, and
        values are the keys for the items in data that go into the split.

    Returns
    -------
    split_dataset : dict
        The split dataset.
    """
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

# def save_database()


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


def get_split_data(molecules, splits):
    
    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    
    trn_mols = [molecules[i] for i in splits['train']]
    val_mols = [molecules[i] for i in splits['valid']]
    tst_mols = [molecules[i] for i in splits['test']]
    
    # Convert list-of-dicts to dict-of-lists
    trn_mols = {prop: [mol[prop] for mol in trn_mols] for prop in props}
    val_mols = {prop: [mol[prop] for mol in val_mols] for prop in props}
    tst_mols = {prop: [mol[prop] for mol in tst_mols] for prop in props}
        

    gdb9_data = {}
    
    # train
    mols = {}
    for key, val in trn_mols.items():
        if key in {'adj', 'attn_bias', 'spatial_pos'}:
            # Determine the maximum shape of the tensors in the list
            max_shape = max([t.shape for t in val])

            # Pad the tensors in the list to the maximum shape
            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])


        elif key in {'edge_attr'}:
            max_shape = max([t.shape[:2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        elif key in {'edge_input'}:
            max_shape = max([t.shape[:2] for t in val])
            max_dist = max([t.shape[2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_dist - t.shape[2], 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        else:
            if val[0].dim() > 0:
                mols[key] = pad_sequence(val, batch_first=True)
            else:
                mols[key] = torch.stack(val)


    assert ((mols['adj'] != mols['edge_attr'][:, :, :, 1:].sum(-1)).sum() == 0), 'sum(adj) != sum(edge_attr)'
    gdb9_data['train'] = mols
    
    
    # val
    mols = {}
    for key, val in val_mols.items():
        if key in {'adj', 'attn_bias', 'spatial_pos'}:
            # Determine the maximum shape of the tensors in the list
            max_shape = max([t.shape for t in val])

            # Pad the tensors in the list to the maximum shape
            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])


        elif key in {'edge_attr'}:
            max_shape = max([t.shape[:2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        elif key in {'edge_input'}:
            max_shape = max([t.shape[:2] for t in val])
            max_dist = max([t.shape[2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_dist - t.shape[2], 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        else:
            if val[0].dim() > 0:
                mols[key] = pad_sequence(val, batch_first=True)
            else:
                mols[key] = torch.stack(val)


    assert ((mols['adj'] != mols['edge_attr'][:, :, :, 1:].sum(-1)).sum() == 0), 'sum(adj) != sum(edge_attr)'
    gdb9_data['valid'] = mols
    
    
    # test
    mols = {}
    for key, val in tst_mols.items():
        if key in {'adj', 'attn_bias', 'spatial_pos'}:
            # Determine the maximum shape of the tensors in the list
            max_shape = max([t.shape for t in val])

            # Pad the tensors in the list to the maximum shape
            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])


        elif key in {'edge_attr'}:
            max_shape = max([t.shape[:2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        elif key in {'edge_input'}:
            max_shape = max([t.shape[:2] for t in val])
            max_dist = max([t.shape[2] for t in val])

            mols[key] = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_dist - t.shape[2], 0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0])) for t in val])

        else:
            if val[0].dim() > 0:
                mols[key] = pad_sequence(val, batch_first=True)
            else:
                mols[key] = torch.stack(val)


    assert ((mols['adj'] != mols['edge_attr'][:, :, :, 1:].sum(-1)).sum() == 0), 'sum(adj) != sum(edge_attr)'
    gdb9_data['test'] = mols
                
    return gdb9_data
    

def process_qdb9_qm9(data, property_file, file_idx_list=None, stack=True):
    
    with open(property_file, 'r') as f:
        target = f.read().split('\n')[1:-1]
        target = [[float(x) for x in line.split(',')[1:20]]
                  for line in target]
        target = torch.tensor(target, dtype=torch.float)
        
#     with open(exclude_file, 'r') as f:
#         skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
    
    suppl = Chem.SDMolSupplier(data, removeHs=False,sanitize=False)

    molecules = read_from_qdb9(suppl, target)
        

    return molecules


def read_from_qdb9(suppl, target):

    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}
    charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    data_list = []
    for i, (mol, tar) in enumerate(tqdm(zip(suppl, target))):

        N = mol.GetNumAtoms()

        conf = mol.GetConformer()
        pos = conf.GetPositions()
        atom_positions = torch.tensor(pos, dtype=torch.float)

        atom_charges = []

        for atom in mol.GetAtoms():
            atom_charges.append(charge_dict[atom.GetSymbol()])

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = torch.nn.functional.one_hot(edge_type, num_classes=len(bonds))

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]
        
        adj = to_dense_adj(edge_index)[0].long()
        adj_comp = 1 - adj
        
        attn_edge_type = torch.zeros([N, N, len(bonds)+1], dtype=torch.long)
        attn_edge_type[:, :, 1:][adj != 0] = edge_attr
        attn_edge_type[:, :, 0] = adj_comp
        num_atoms = torch.tensor(N)
        atom_charges = torch.tensor(atom_charges)
        index = torch.tensor(i)

        #shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        #max_dist = np.amax(shortest_path_result)
        #spatial_pos = torch.from_numpy((shortest_path_result)).long()
        #edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())+1
        #edge_input = torch.from_numpy(edge_input).long()
        #attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        mol_prop = {'A': target[i][0], 'B': target[i][1], 'C': target[i][2], 'mu': target[i][3], 'alpha': target[i][4], 'homo': target[i][5], 'lumo': target[i][6], 'gap': target[i][7], 'r2': target[i][8], 'zpve': target[i][9], 'U0': target[i][10], 'U': target[i][11], 'H': target[i][12], 'G': target[i][13], 'Cv': target[i][14]}
        
        molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions, 'index': index, 'adj': adj, 'edge_attr': attn_edge_type}
        
        #graph = {'attn_bias': attn_bias, 'spatial_pos': spatial_pos, 'edge_input': edge_input}
        #graph = {'spatial_pos': spatial_pos, 'edge_input': edge_input}
        
        molecule.update(mol_prop)
        #molecule.update(graph)
    
        data_list.append(molecule)
    
    return data_list


def process_xyz_md17(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the MD-17 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    line_counter = 0
    atom_positions = []
    atom_types = []
    for line in xyz_lines:
        if line[0] == '#':
            continue
        if line_counter == 0:
            num_atoms = int(line)
        elif line_counter == 1:
            split = line.split(';')
            assert (len(split) == 1 or len(split) == 2), 'Improperly formatted energy/force line.'
            if (len(split) == 1):
                e = split[0]
                f = None
            elif (len(split) == 2):
                e, f = split
                f = f.split('],[')
                atom_energy = float(e)
                atom_forces = [[float(x.strip('[]\n')) for x in force.split(',')] for force in f]
        else:
            split = line.split()
            if len(split) == 4:
                type, x, y, z = split
                atom_types.append(split[0])
                atom_positions.append([float(x) for x in split[1:]])
            else:
                logging.debug(line)
        line_counter += 1

    atom_charges = [charge_dict[type] for type in atom_types]

    molecule = {'num_atoms': num_atoms, 'energy': atom_energy, 'charges': atom_charges,
                'forces': atom_forces, 'positions': atom_positions}

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    
    
    
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]
    mol_smiles = xyz_lines[num_atoms+3]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])
        
#     mol = mol_smiles.split()[-1]
#     mol = Chem.MolFromSmiles(mol)
#     if mol is None:
#         print(mol_smiles)
    
#     N = mol.GetNumAtoms()
#     type_idx = []
#     for atom in mol.GetAtoms():
#         type_idx.append(types[atom.GetSymbol()])
    
#     row, col, edge_type = [], [], []
#     for bond in mol.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_type += 2 * [bonds[bond.GetBondType()] + 1]
        
#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_type = torch.tensor(edge_type, dtype=torch.long)
#     edge_attr = torch.nn.functional.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
    
#     perm = (edge_index[0] * N + edge_index[1]).argsort()
#     edge_index = edge_index[:, perm]
#     edge_attr = edge_attr[perm]
    
#     type_idx = torch.Tensor(type_idx).long()
#     to_keep = type_idx > 0
#     edge_index_wo_h, edge_attr_wo_h = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True, num_nodes=len(to_keep))

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    #molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions, 'edge_index': edge_index, 'edge_attr': edge_attr, 'edge_index_wo_h': edge_index_wo_h, 'edge_attr_wo_h': edge_attr_wo_h}
    
    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}
    
    return molecule
