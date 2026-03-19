import glob
import torch
import os

from tqdm import tqdm
from torch_geometric.data import Data

from utils.processing import load_mat_file, get_node_features, get_edge_attrY, damages_from_features

def process_zhu_n_minus_k(cfg):
    """
    Process n-k contingency simulation data from .mat files and save as PyTorch geometric Data objects.
    This function loads MATLAB simulation files containing power grid n-k contingency scenarios,
    extracts and processes node and edge features, computes voltage magnitudes, and saves the
    processed graph data as PyTorch geometric Data objects.
    Args:
        cfg: ProcessingConfiguration object containing:
            - processed_dir (str): Directory path where processed .pt files will be saved
    Returns:
        None. Saves processed PyTorch geometric Data objects to cfg.processed_dir with 
        filenames in the format 'data_{scenario}.pt'
    Notes:
        - Only processes simulations marked as successful (success flag == True)
        - Extracts node features including real/imaginary power and voltage magnitude
        - Creates graph representation with adjacency matrix and edge attributes from admittance data
        - Skips scenarios with unsuccessful simulations with a console message
    Raises:
        FileNotFoundError: If no simulation*.mat files are found in the 'raw/' directory
    """

    processed_dir = cfg.processed_dir
    # Get all .mat files starting with 'simulation'
    mat_files = sorted(glob.glob("raw/simulation*.mat"))

    for file in tqdm(mat_files, desc="processing raw n-k simulations"):
        scenario = str(file).split('_')[1].split('.')[0]

        simulation_data, filetype = load_mat_file(file)
        simulation_data = simulation_data['simulation'][0][0]
        perturbed_network = simulation_data['perturbed_network'][0][0]
        final_network = simulation_data['final_network'][0][0]

        success = final_network['success'][0][0]

        if not success:
            print(f"Not saving scenario {scenario} because simulation was unsuccessful")
            continue

        init_data = perturbed_network

        node_data_pre = perturbed_network[2]
        gen_data_pre = perturbed_network[3]
        edge_data_pre = simulation_data['initialY'][0][0][0].toarray() #convert from sparse for simplicity
        edge_IDs = perturbed_network[4][:,:2]

        node_data_post = final_network[2]
        gen_data_post = final_network[3]
        edge_data_post = final_network[11].toarray() #y matrix converted from sparse

        node_feature, node_labels, graph_label = get_node_features(node_data_pre, node_data_post, gen_data_pre,
                                                                        gen_data_post=gen_data_post)  # extract node features and labels from data


        #load normal (not Y) edge features to find inactive edges
        decoded_damages = damages_from_features(perturbed_network['branch'])

        adj, edge_attr = get_edge_attrY(edge_data_pre, decoded_damages)

        admittance_matrix = torch.tensor(edge_data_post)

        #
        Vreal = node_feature[:, 2]
        Vimag = node_feature[:, 3]
        Vm = torch.sqrt(Vreal ** 2 + Vimag ** 2).unsqueeze(1)

        node_feature = torch.cat([
            node_feature[:, 0:2],  # S real and imag
            Vm,  # only magnitude used as a feature in Zhu paper
            node_feature[:, 4:7],  # inactive buses not in features
        ], dim=1)

        data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr, node_labels=node_labels[:, :2],
                    admittance_matrix=admittance_matrix, y=graph_label)


        torch.save(data, os.path.join(processed_dir, f'data_{scenario}.pt'))



    