# Atom definition
atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}

n_nodes_with_H = [0, 0, 0, 1, 2, 1, 13, 20, 115, 227, 803, 1478, 4820, 7995, 18985, 24113, 37286, 23933, 11133, 2388, 181, 10, 1, 1, 1, 1, 1, 1, 1, 1]
n_nodes_without_H = [0, 0, 0, 2, 5, 26, 135, 928, 6994, 121910]

atom_weights_with_H = {0: 1.008, 1: 12.01, 2: 14.01, 3: 16.00, 4: 19.00}
atom_weights_without_H = {0: 12.01, 1: 14.01, 2: 16.00, 3: 19.00}


# QM9 dataset
qm9_with_h = {
    'name': 'qm9_with_h',
    'atom_encoder': atom_encoder,
    'atom_decoder': atom_decoder,
    'num_species': len(atom_decoder),
    'charge_scale': 1,
    'n_nodes': n_nodes_with_H,
    'max_n_nodes': len(n_nodes_with_H) - 1,
    'max_weight': 0,
    'edge_types': [1, 1, 1, 1],
    'atom_weights': atom_weights_with_H,
    'max_in_deg': 32,
    'max_out_deg': 32,
    'max_num_edges': 50,
    'max_spatial': 50,
    'max_edge_dist': 10
}

qm9_without_h = {
    'name': 'qm9_without_h',
    'atom_encoder': {atom: i for i, atom in enumerate(atom_decoder[1:])},
    'atom_decoder': atom_decoder[1:],
    'num_species': len(atom_decoder) - 1,
    'charge_scale': 1,
    'n_nodes': n_nodes_without_H,
    'max_n_nodes': len(n_nodes_without_H) - 1,
    'max_weight': 0,
    'edge_types': [1, 1, 1, 1],
    'atom_weights': atom_weights_without_H,
    'max_in_deg': 32,
    'max_out_deg': 32,
    'max_num_edges': 50,
    'max_spatial': 50,
    'max_edge_dist': 10
}

# Geom dataset
geom_with_h = {
    'name': 'geom_with_h',
    'atom_encoder': {
        'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'B': 5, 'Al': 6, 'Si': 7,
        'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15
    },
    'atom_decoder': [
        'H', 'C', 'N', 'O', 'F', 'B', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'
    ],
    'num_species': 16,
    'charge_scale': 1,
    'n_nodes': n_nodes_with_H,  # Using QM9 with H as a placeholder
    'max_n_nodes': len(n_nodes_with_H) - 1,
    'max_weight': 0,
    'edge_types': [1, 1, 1, 1],
    'atom_weights': {i: 1 for i in range(16)},  # Placeholder
    'max_in_deg': 32,
    'max_out_deg': 32,
    'max_num_edges': 50,
    'max_spatial': 50,
    'max_edge_dist': 10
}


def get_dataset_info(dataset_name, remove_h):
    if dataset_name == 'qm9':
        if remove_h:
            return qm9_without_h
        else:
            return qm9_with_h
    elif dataset_name == 'geom':
        return geom_with_h
    raise Exception("Unknown dataset {}".format(dataset_name))
