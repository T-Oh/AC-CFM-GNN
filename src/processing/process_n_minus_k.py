import torch
import scipy.io
import os

from torch_geometric.data import Data

from utils.processing import get_initial_damages, get_scenario_of_file
from utils.processing import get_edge_features, get_node_features  
def process_n_minus_k(cfg):
    """
    Loads the raw matlab data, converts it to torch
    tensors which are then saved. Nees to be manually normalized after processing using normalize.py
    """
    #INIT
    #load scenario file which stores the initial damages
    damages = get_initial_damages()
    #load initial network data
    init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
    #Initial data
    node_data_pre = init_data['ans'][0,0][2]    #ans is correct bcs its pwsdata
    gen_data_pre = init_data['ans'][0,0][3]
    edge_data = init_data['ans'][0,0][4]
    below_threshold_count = 0

    #PROCESSING
    for raw_path in cfg.raw_paths:
        #skip damage file and pws file 
        if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
            continue
        scenario = get_scenario_of_file(raw_path)
        file=scipy.io.loadmat(raw_path)  #loads a full scenario 

        accumulated_ls_tot = 0.  #used for step selection according to ls_threshold and N_below_threshold
        remaining_load = 1.      #necessary to correctly scale ls_tot since it refers to load shed relative to initial load at each step (not init load of scenario)

        #loop through steps of scenario each step will be one processed data file
        for i in range(min(len(file['clusterresult'][0,:]), 10)):
            accumulated_ls_tot += file['clusterresult'][0,i][21] * remaining_load
            remaining_load -= file['clusterresult'][0,i][21] * remaining_load

            #skip if total loadshed of timestep is below threshold and the amount of low loadshed instances is reached
            if cfg.data_type == 'LSTM' or accumulated_ls_tot>cfg.ls_threshold or below_threshold_count<cfg.N_below_threshold:
                if below_threshold_count<cfg.N_below_threshold and accumulated_ls_tot<cfg.ls_threshold:
                    below_threshold_count += 1

                if np.isnan(file['clusterresult'][0,i][21]):    #This refers to matlab column ls_total -> if this is NaN the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                    print('Skipping because ls_tot==NaN', file, i)
                    continue
                node_data_post = file['clusterresult'][0,i][2]   #node_data after step i for node_label_calculation

                node_feature, node_labels = get_node_features(node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data                    

                adj, edge_attr = get_edge_features(edge_data, damages, node_data_pre, scenario, i, n_minus_k=True)



                graph_label = node_labels.sum()
                
                data = Data(x=node_feature.float(), edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1), node_labels=node_labels, y=graph_label) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
    
