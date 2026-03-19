
import numpy as np
import torch
import os

from torch_geometric.data import Data

from utils.processing import ProcessingConfig
from utils.processing import get_initial_damages, get_scenario_of_file, load_mat_file, get_ls_tot, get_data, save_static_data, decode_damage
from utils.processing import get_edge_features, get_node_features, get_edge_labels, get_edge_attrY_Zhumat73, get_edge_attrY
from utils.utils import check_s_y_relation
def process_ac(cfg: ProcessingConfig):
    """
    Loads the raw matlab data, converts it to torch
    tensors which are then saved. Then the torch data is reloaded and normalized 
    - this is because loading the matfiles takes extremely long as there is much
    more data saved there so I want to avoid loading them twice

    """
    #INIT
    #load scenario file which stores the initial damages
    damages = get_initial_damages()
    KEY = 'clusterresult_'
    #load initial network data
    init_data, filetype = load_mat_file('raw/' + 'pwsdata.mat')
    #For LSTM we add a single file that is the static solution with no damages which will be used to pad the sequences
    if cfg.data_type == 'LSTM':
        adj_init = save_static_data(cfg, cfg.processed_dir, KEY, init_data, filetype)

    below_threshold_count = 0


    #PROCESSING
    for raw_path in cfg.raw_paths:
        #skip damage file and pws file 
        if 'Hurricane' in raw_path or 'pwsdata' in raw_path:    continue

        #get scenario ID
        scenario = get_scenario_of_file(raw_path)

        #load file
        file, filetype = load_mat_file(raw_path)  #loads a full scenario 
        if not filetype == 'LOAD_FAILED':
            if 'clusterresult_' not in file:
                file['clusterresult_'] = file['pf_result']

        #get total loadshed of each step
            len_scenario, ls_tot = get_ls_tot(KEY, filetype, file)

            #initialize variable for cummulative loadshed
            if cfg.data_type == 'LSTM':    cummulative_ls = 0

            #Loop through all steps of the scenario
            for i in range(len_scenario):
                #skip if total loadshed of timestep is below threshold and the amount of low loadshed instances is reached
                if cfg.data_type == 'LSTM' or ls_tot[i]>cfg.ls_threshold or below_threshold_count<cfg.N_below_threshold:
                    #adjust below_threshold_count
                    if below_threshold_count<cfg.N_below_threshold and ls_tot[i]<cfg.ls_threshold:    below_threshold_count += 1

                    #skip step if ls_tot is NaN
                    if np.isnan(ls_tot[i]):                #This refers to matlab column ls_total -> if this is NaN the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                        print('Skipping', file, i, 'because ls_tot==NaN')
                        if cfg.data_type == 'LSTM':    break
                        else:                           continue

                    #extract necessary data
                    if i == 0:  node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, edge_IDs = get_data(cfg, init_data, file, KEY, i, filetype)                    
                    else:       node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, _ = get_data(cfg, init_data, file, KEY, i, filetype)                    

                    #extract node features and labels from data
                    node_feature, node_labels, graph_label = get_node_features(cfg, node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data  
      

                #extract edge features from data
                if cfg.edge_attr_type == 'Y':
                    decoded_damages = decode_damage(damages[scenario], i, node_data_pre[:,0], edge_IDs)
                    if i!=0 and filetype == 'Zhu_mat73':  
                        adj, edge_attr = get_edge_attrY_Zhumat73(edge_data_pre, decoded_damages)
                        if cfg.data_type == 'LSTM':    adj_post, edge_attr_post = get_edge_attrY_Zhumat73(edge_data_post, [])
                    else:    
                        adj, edge_attr = get_edge_attrY(edge_data_pre, decoded_damages)
                        if cfg.data_type == 'LSTM':    adj_post, edge_attr_post = get_edge_attrY(edge_data_post, [])
                    if cfg.check_s_y:
                        check_s_y_relation(node_data_post, edge_data_post, gen_data_post)
                else:
                    adj, edge_attr = get_edge_features(edge_data_pre, damages, node_data_pre, scenario, i, n_minus_k=False)
                
                #save unscaled data (non LSTM)
                if cfg.data_type in ['AC', 'ANGF_Vcf']:
                    data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
                    torch.save(data, os.path.join(cfg.processed_dir, f'data_{scenario}_{i}.pt'))
                elif cfg.data_type in ['Zhu', 'Zhu_mat73', 'Zhu_nobustype']:
                    data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr, node_labels=node_labels[:,:2], y=graph_label)
                    torch.save(data, os.path.join(cfg.processed_dir, f'data_{str(scenario)}_{str(i)}.pt'))
                    
                
                if cfg.data_type == 'LSTM':
                    cummulative_ls += graph_label
                    edge_labels = get_edge_labels(adj_init, adj_post, edge_attr_post)
                    data = Data(x=node_feature.to(torch.float32), edge_index=adj, edge_attr=edge_attr.to(torch.float32), node_labels=node_labels.to(torch.float32), 
                                y=graph_label.to(torch.float32), y_cummulative=torch.as_tensor(cummulative_ls).to(torch.float32), 
                                edge_labels=torch.as_tensor(edge_labels).to(torch.float32)) 
                    scenario_dir = os.path.join(cfg.root, f'processed/scenario_{scenario}')
                    os.makedirs(scenario_dir, exist_ok=True)
                    torch.save(data, os.path.join(scenario_dir, f'data_{scenario}_{i}.pt'))
