"""
Author : Tobias Ohlinger

File setting up training and test data as well as neural network models
"""
import os
import numpy as np
import scipy.io
import time
import torch
import json
import h5py
from os.path import isfile

from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader
from torch.utils.data import Subset, random_split
from torch_geometric.utils import to_undirected
from torch.nn.utils.rnn import pad_sequence
import torch.utils
from functools import partial






#######################

# Custom datasets

class HurricaneDataset(Dataset):
    """
    Custom class for the hurricane dataset
    use_supernode   deprecated
    transform       deprecated
    pre_transform   deprecated
    pre_filter      deprecated
    N_Scenarios     the number of the last scenario independent of the actual number of used scenarios
    stormsplit      no stormsplit is applied if 0 otherwise the data is split by putting all instances where the scenario indicator starts with stormsplit (f.e. 1) in the test set
    embedding       Node2Vec embedding to be used
    device          torch device
    data_type       'AC' or 'DC' (only intended for AC)
    """

    
    def __init__(self, root, use_supernode, transform=None, pre_transform=None, pre_filter=None, N_Scenarios=100, stormsplit=0, embedding=None, device=None, data_type='AC', edge_attr='multi', ls_threshold = .09, N_below_threshold=1):
        self.use_supernode=use_supernode
        self.embedding = embedding
        self.device = device
        self.data_type = data_type
        self.edge_attr = edge_attr
        self.ls_threshold = ls_threshold
        self.N_below_threshold = N_below_threshold
        super().__init__(root, transform, pre_transform, pre_filter)
        self.stormsplit = stormsplit
        self.data_list = self.processed_file_names
        #self.data_list=self.get_data_list(N_Scenarios)  #list containing all instances in order

        
        
    
    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = []
        for root, _, filenames in os.walk(self.root + "/processed"):
            for filename in filenames:
                if filename.startswith("data"):
                    files.append(os.path.relpath(os.path.join(root, filename), self.root + "/processed"))
        return files
        
    
    def get_data_list(self,N_scenarios):
        #test_id is the id given to the storm when compiling the dataset of all storms (i.e. the first digit of the scenario (f.e. Claudette=1)) and is used to relate the data files to the storms
        #To use the percentage based train test split stormsplit should be set to 0
        #N_scenario must be last Scenario that appears in raw (if scenario 1,2 and 100 are used N_scenarios must be 100)
        data_list=np.zeros((len(self.processed_file_names),2))
        idx=0  
        test_idx=0 
        #Regular Split 
        if self.stormsplit == 0:   
            for file in self.processed_file_names:
                if file.startswith('data'):
                    scenario, step = self.get_scenario_step_of_file(file)
                    data_list[idx,:] = [scenario, step]                   
                    idx += 1       
        #Stormsplit
        else:   
            test_idx = len(data_list)-1
            for file in self.processed_file_names:           
                if file.startswith('data'):
                    scenario, step = self.get_scenario_step_of_file(file)
                    if str(scenario).startswith(str(self.stormsplit)):
                        data_list[test_idx,:] = [scenario, step]
                        test_idx -= 1
                    else:
                        data_list[idx,:] = [scenario, step]
                        idx += 1
            
        return data_list
    
    def process(self):
        if self.data_type in ['AC', 'LSTM', 'Zhu', 'Zhu_mat73', 'ANGF_Vcf', 'Zhu_nobustype']:
            self.process_ac()
        elif self.data_type == 'LDTSF':
            self.process_ldtsf()
        elif self.data_type == 'LDTSF_DC':
            self.process_ldtsf_dc()
        elif self.data_type == 'n-k':
            self.process_n_minus_k()
        elif self.data_type == 'DC':
            self.process_dc()
        else:
            assert False, 'Datatype must be AC or DC!'

    def process_ac(self):
        """
        Loads the raw matlab data, converts it to torch
        tensors which are then saved. Then the torch data is reloaded and normalized 
        - this is because loading the matfiles takes extremely long as there is much
        more data saved there so I want to avoid loading them twice

        """
        #INIT
        #load scenario file which stores the initial damages
        damages = self.get_initial_damages()
        KEY = 'clusterresult_'
        #load initial network data
        init_data, filetype = self.load_mat_file('raw/' + 'pwsdata.mat')
        #For LSTM we add a single file that is the static solution with no damages which will be used to pad the sequences
        if self.data_type == 'LSTM':
            adj_init = self.save_static_data(KEY, init_data, filetype)

        below_threshold_count = 0


        #PROCESSING
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:    continue

            #get scenario ID
            scenario = self.get_scenario_of_file(raw_path)

            #load file
            file, filetype = self.load_mat_file(raw_path)  #loads a full scenario 

            #get total loadshed of each step
            len_scenario, ls_tot = self.get_ls_tot(KEY, filetype, file)

            #initialize variable for cummulative loadshed
            if self.data_type == 'LSTM':    cummulative_ls = 0

            #Loop through all steps of the scenario
            for i in range(len_scenario):
                #skip if total loadshed of timestep is below threshold and the amount of low loadshed instances is reached
                if self.data_type == 'LSTM' or ls_tot[i]>self.ls_threshold or below_threshold_count<self.N_below_threshold:
                    #adjust below_threshold_count
                    if below_threshold_count<self.N_below_threshold and ls_tot[i]<self.ls_threshold:    below_threshold_count += 1

                    #skip step if ls_tot is NaN
                    if np.isnan(ls_tot[i]):                #This refers to matlab column ls_total -> if this is NaN the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                        print('Skipping', file, i, 'because ls_tot==NaN')
                        if self.data_type == 'LSTM':    break
                        else:                           continue

                    #extract necessary data
                    if i == 0:  node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, edge_IDs = self.get_data(init_data, file, KEY, i, filetype)                    
                    else:       node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, _ = self.get_data(init_data, file, KEY, i, filetype)                    

                    #extract node features and labels from data
                    node_feature, node_labels, graph_label = self.get_node_features(node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data  

                    #extract edge features from data
                    if self.edge_attr == 'Y':
                        decoded_damages = self.decode_damage(damages[scenario], i, node_data_pre[:,0], edge_IDs)
                        if i!=0 and filetype == 'Zhu_mat73':  
                            adj, edge_attr = self.get_edge_attrY_Zhumat73(edge_data_pre, decoded_damages)
                            if self.data_type == 'LSTM':    adj_post, edge_attr_post = self.get_edge_attrY_Zhumat73(edge_data_post, [])
                        else:    
                            adj, edge_attr = self.get_edge_attrY(edge_data_pre, decoded_damages)
                            if self.data_type == 'LSTM':    adj_post, edge_attr_post = self.get_edge_attrY(edge_data_post, [])
                    else: 
                        adj, edge_attr = self.get_edge_features(edge_data_pre, damages, node_data_pre, scenario, i, n_minus_k=False)
                    
                    #save unscaled data (non LSTM)
                    if self.data_type in ['AC', 'ANGF_Vcf']:
                        data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
                        torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
                    elif self.data_type in ['Zhu', 'Zhu_mat73', 'Zhu_nobustype']:
                        data = Data(x=node_feature, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels[:,:2], y=graph_label) 
                        torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
                        
                    
                if self.data_type == 'LSTM':
                    cummulative_ls += graph_label
                    edge_labels = self.get_edge_labels(adj_init, adj_post, edge_attr_post)
                    data = Data(x=node_feature.to(torch.float32), edge_index=adj, edge_attr=edge_attr.to(torch.float32), node_labels=node_labels.to(torch.float32), 
                                y=graph_label.to(torch.float32), y_cummulative=torch.tensor(cummulative_ls).to(torch.float32), 
                                edge_labels=torch.tensor(edge_labels).to(torch.float32)) 
                    scenario_dir = os.path.join(self.root, f'processed/scenario_{scenario}')
                    os.makedirs(scenario_dir, exist_ok=True)
                    torch.save(data, os.path.join(scenario_dir, f'data_{scenario}_{i}.pt'))

    def get_ls_tot(self, KEY, filetype, file):
        if filetype == 'Zhu_mat73':
            ls_tot_ref = file[KEY]['ls_total']
            len_scenario = len(ls_tot_ref)              
            ls_tot = []
            for step in range(len_scenario):
                ls_tot_deref = file[ls_tot_ref[step,0]]
                ls_tot.append(ls_tot_deref[()])
        else:
            len_scenario = len(file[KEY][0,:])
            ls_tot = []
            for step in range(len_scenario):
                ls_tot.append(file[KEY][0,step][21])
        return len_scenario,ls_tot

    def save_static_data(self, KEY, init_data, filetype):
        assert self.edge_attr == 'Y', 'Edge attribute must be Y for static data' 
        node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, _ = self.get_data(init_data, init_data, KEY, -1, filetype)                    
        node_feature, node_labels, graph_label = self.get_node_features(node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data  

        adj, edge_attr = self.get_edge_attrY(edge_data_pre, [])
        edge_labels = self.get_edge_labels(adj, adj, edge_attr)

        data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr.float(), node_labels=node_labels.float(), y=graph_label.float(), 
                    y_cummulative=torch.tensor(0).to(torch.float32), edge_labels=edge_labels.to(torch.float32))
        torch.save(data, os.path.join(self.processed_dir, f'data_static.pt'))
        #returns the adjacency matrix of the static data which is used to determine the edge labels in the LSTM data
        return adj



    def get_data(self, init_data, file, KEY, i, filetype):
        if i <= 0:  #in first iteration load original pwsdata as initial data 
            node_data_pre = init_data[KEY][0,0][2] 
            gen_data_pre = init_data[KEY][0,0][3]
            if self.edge_attr == 'Y':                           
                edge_data_pre = init_data[KEY][0,0][10]    #loading the added Admittance matrix instead of the edge data
                edge_IDs = init_data[KEY][0,0][4][:,:2]
            else: 
                edge_data_pre = init_data[KEY][0,0][4]

            if filetype == 'Zhu_mat73':           #Zhu_mat73 is only necessary for Ike and Harvey where the files were too big and need to be saved in the newer mat7.3 format
                node_data_post = []
                bus_data_ref = file[KEY]['bus']
                ref = bus_data_ref[i, 0]  # Get the object reference
                dereferenced_data = file[ref]  # Dereference it
                node_data_post.append(dereferenced_data[()])  # Append the actual data
                node_data_post = torch.tensor(np.array(node_data_post).squeeze()).transpose(0,1)  

                edge_data_post = []
                edge_data_ref = file[KEY]['Ybus_ext']
                ref = edge_data_ref[i,0]
                dereferenced_data = file[ref]
                edge_data_post.append(dereferenced_data[()])          
            else:
                node_data_post = file[KEY][0,i][2]   #node_data after step i for node_label_calculation
                if i == -1: edge_data_post = edge_data_pre
                else:       
                    edge_data_post = file[KEY][0,i][29][0]   #edge data after step i for edge_label_calculation
                

        else:
            edge_IDs = None
            if filetype == 'Zhu_mat73':
                node_data_pre = []
                bus_data_ref = file[KEY]['bus']
                ref = bus_data_ref[i-1, 0]
                dereferenced_data = file[ref]
                node_data_pre.append(dereferenced_data[()])

                gen_data_pre = []
                gen_data_ref = file[KEY]['gen']
                ref = gen_data_ref[i-1,0]
                dereferenced_data = file[ref]
                gen_data_pre.append(dereferenced_data[()])

                edge_data_pre = []
                edge_data_ref = file[KEY]['Ybus_ext']
                ref = edge_data_ref[i-1,0]
                dereferenced_data = file[ref]
                edge_data_pre.append(dereferenced_data[()])

                node_data_post = []
                bus_data_ref = file[KEY]['bus']
                ref = bus_data_ref[i, 0]  # Get the object reference
                dereferenced_data = file[ref]  # Dereference it
                node_data_post.append(dereferenced_data[()])  # Append the actual data

                edge_data_post = []
                edge_data_ref = file[KEY]['Ybus_ext']
                ref = edge_data_ref[i,0]
                dereferenced_data = file[ref]
                edge_data_post.append(dereferenced_data[()])

                node_data_pre = torch.tensor(np.array(node_data_pre).squeeze()).transpose(0,1)
                node_data_post = torch.tensor(np.array(node_data_post).squeeze()).transpose(0,1)

                # Convert edge_data to a NumPy array for processing
                edge_data_pre_array = np.array(edge_data_pre)
                edge_data_post_array = np.array(edge_data_post)

                # Check if 'dtype' exists and whether it has named fields
                if hasattr(edge_data_pre_array, 'dtype') and edge_data_pre_array.dtype.names:
                    # Extract real and imaginary parts
                    real_part_pre = edge_data_pre_array['real'].squeeze()
                    imag_part_pre = edge_data_pre_array['imag'].squeeze()
                else:
                    # No dtype field, treat the entire array as the real part
                    real_part_pre = edge_data_pre_array.squeeze()
                    imag_part_pre = np.zeros_like(real_part_pre)

                # Check if 'dtype' exists and whether it has named fields
                if hasattr(edge_data_post_array, 'dtype') and edge_data_post_array.dtype.names:
                    # Extract real and imaginary parts
                    real_part_post = edge_data_post_array['real'].squeeze()
                    imag_part_post = edge_data_post_array['imag'].squeeze()
                else:
                    # No dtype field, treat the entire array as the real part
                    real_part_post = edge_data_post_array.squeeze()
                    imag_part_post = np.zeros_like(real_part_post)

                # Create the complex tensors
                edge_data_pre = torch.complex(torch.tensor(real_part_pre), torch.tensor(imag_part_pre))
                edge_data_post = torch.complex(torch.tensor(real_part_post), torch.tensor(imag_part_post))

                gen_data_pre = torch.tensor(np.array(gen_data_pre).squeeze()).transpose(0,1)

            else:
                node_data_pre = []
                gen_data_pre = []
                edge_data_pre = []
                edge_data_post = []
                
                node_data_pre = file[KEY][0,i-1][2]    #node_data of initial condition of step i
                gen_data_pre = file[KEY][0,i-1][3]
                if self.edge_attr == 'Y':                           
                    edge_data_pre = file[KEY][0,i-1][29]       #loading the added Admittance matrix instead of the edge data
                    edge_data_post = file[KEY][0,i][29]       #loading the added Admittance matrix instead of the edge data
                else:
                    edge_data_pre = file[KEY][0,i-1][4]         #edge data of initial condition of step i
                node_data_post = file[KEY][0,i][2]   #node_data after step i for node_label_calculation
            
            
        return node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, edge_IDs

           
    def process_ldtsf(self):
        '''
        Processes the data so that the input is the sequence of initial damages and the output is the total load shed of each scenario
        '''
        damages = self.get_initial_damages()
        KEY = 'clusterresult_'

        init_data, filetype = self.load_mat_file('raw/' + 'pwsdata.mat')
        edge_data = init_data[KEY][0,0][4]
        bus_from = edge_data[:,0]
        bus_to = edge_data[:,1]
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
                continue
            scenario = self.get_scenario_of_file(raw_path)
            file, filetype = self.load_mat_file(raw_path)  #loads a full scenario 
            #_, _, edge_data, _, _ = self.get_data(file[KEY][0,0][4]
            
            #get total loadshed of each step
            N_steps, ls_tot = self.get_ls_tot(KEY, filetype, file)
            #N_steps = len(file[KEY][0,:])
            N_damages = len(damages[scenario][:,0])
            scenario = self.get_scenario_of_file(raw_path)

            x = torch.zeros([len(damages[scenario][:,1]),3206])
            #for i in range(len(damages[scenario][:,1])):
                

            #add parallel lines to the damaged lines to x
            dmg_idx = 0
            shifted_step = 0
            while dmg_idx < N_damages:
                x[shifted_step,damages[scenario][dmg_idx,1]-1] = 1    #sequence of initial damages
                j=0
                while bus_from[damages[scenario][dmg_idx,1]-1] == bus_from[damages[scenario][dmg_idx,1]-1-j] and bus_to[damages[scenario][dmg_idx,1]-1] == bus_to[damages[scenario][dmg_idx,1]-1-j]:
                    x[shifted_step][damages[scenario][dmg_idx,1]-1-j] = 1 
                    j = j+1
                j = 0
                while bus_from[damages[scenario][dmg_idx,1]-1] == bus_from[damages[scenario][dmg_idx,1]-1+j] and bus_to[damages[scenario][dmg_idx,1]-1] == bus_to[damages[scenario][dmg_idx,1]-1+j]:
                    x[shifted_step][damages[scenario][dmg_idx,1]-1+j] = 1 
                    j = j+1

                #check if there is another damage in the same time step
                while dmg_idx < len(damages[scenario])-1 and damages[scenario][dmg_idx,0] == damages[scenario][dmg_idx+1,0]:  
                    x[shifted_step][damages[scenario][dmg_idx+1,1]-1] = 1    
                    dmg_idx = dmg_idx+1  
                dmg_idx = dmg_idx+1
                shifted_step = shifted_step+1
            x = x[:shifted_step]
  
            #y = torch.tensor(6.7109e4 - file[KEY][0,-1][17][99]) #17 stores the 'load' array which contains the load after each PF in ACCFM 99 in the last cell of the array containing the final load of this tsep  6.7109e4 is the full load without contingency
            remaining_load = 1
            for i in range(N_steps):
                if i == 0:  y_seq = ls_tot[i]*remaining_load    #file[KEY][0,i][21]*remaining_load
                else:       y_seq = np.append(y_seq, ls_tot[i]*remaining_load)  #np.append(y_seq, file[KEY][0,i][21]*remaining_load)
                remaining_load = remaining_load - ls_tot[i]*remaining_load  #file[KEY][0,i][21]* remaining_load
            y_seq_class = [0 if y_ < 0.15 else 1 for y_ in y_seq]
            y = torch.sum(torch.tensor(y_seq))
            
            if y/6.7109e4 < 0.18:       y_class = 0
            elif y/6.7109e4 < 0.65:     y_class = 1
            elif y/6.7109e4 < 0.88:     y_class = 2
            else:                       y_class = 3

            #y = torch.log(y+1)/torch.log(torch.tensor(6.7109e4+1))    #log normalization
            data = Data(x=x, y=y, y_class=y_class, y_seq=y_seq, y_seq_class=y_seq_class)
            torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{N_steps}.pt'))

    def process_ldtsf_dc(self):
        '''
        Processes the data so that the input is the sequence of initial damages and the output is the total load shed of each scenario
        '''
        init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
        edge_data = init_data['ans'][0,0][4]
        bus_from = edge_data[:,0]
        bus_to = edge_data[:,1]
        scenario = 1 
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
               #used to create unique file identifiers 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
                continue
            #scenario = self.get_scenario_of_file(raw_path)
            with open(raw_path, 'rb') as f:
                data = json.load(f)['result']
            for key in data.keys(): #every file contains 125 scenarios
                damages = self.get_initial_damages_dc(data[key]['primary_dmg'])

                N_steps = len(np.unique(damages[:,0]))

                x = torch.zeros([N_steps,3206])
                i = 0   #index for original damage array (damages in the same time steps appear after one another)
                updated_index = 0   #index for the resulting array where damages in the same time step appear in the same row
                while i < N_steps:
                    x[updated_index,damages[i,1]-1] = 1    #sequence of initial damages
                    #add parallel lines
                    j=0
                    while bus_from[damages[i,1]-1] == bus_from[damages[i,1]-1-j] and bus_to[damages[i,1]-1] == bus_to[damages[i,1]-1-j]:
                        x[updated_index][damages[i,1]-1-j] = 1 
                        j = j+1
                    j = 0
                    while bus_from[damages[i,1]-1] == bus_from[damages[i,1]-1+j] and bus_to[damages[i,1]-1] == bus_to[damages[i,1]-1+j]:
                        x[updated_index][damages[i,1]-1+j] = 1 
                        j = j+1

                    #check if there is another damage in the same time step
                    while i < len(damages[:,0])-1 and damages[i,0] == damages[i+1,0]:  
                        x[updated_index,damages[i+1,1]-1] = 1    
                        i = i+1
                    updated_index = updated_index+1
                    i = i+1

                y = torch.tensor(6.7109e4 -data[key]['final_MW_load']*100)

                
                if y/6.7109e4 < 0.18:       y_class = 0
                elif y/6.7109e4 < 0.65:     y_class = 1
                elif y/6.7109e4 < 0.88:     y_class = 2
                else:                       y_class = 3

                y = torch.log(y+1)/torch.log(torch.tensor(6.7109e4+1))    #log normalization
                processed_data = Data(x=x, y=y, y_class=y_class)
                torch.save(processed_data, os.path.join(self.processed_dir, f'data_{scenario}_{N_steps}.pt'))
                scenario += 1



    def get_edge_labels(self, adj_init, adj_post, edge_attr_post):
        '''
        Calculates the binary edge labels for the LSTM data.
        
        Parameters:
        - adj_init: torch.Tensor (2, E_init) - Initial edge_index
        - adj_post: torch.Tensor (2, E_post) - Updated edge_index
        - edge_attr_post: torch.Tensor (E_post,) - Updated edge attributes (e.g., admittance)
        - threshold: float - Threshold for determining edge labels
        
        Returns:
        - edge_labels: torch.Tensor (E_init,) - Binary labels for edges in adj_init
        '''
        
        threshold = 1e-8
        # Convert edges to set for fast lookup
        adj_post_set = {tuple(edge.tolist()) for edge in adj_post.T}

        # Edge labels
        edge_labels = torch.zeros(adj_init.shape[1], dtype=torch.long)

        # Iterate over edges in the initial adjacency matrix
        for i, edge in enumerate(adj_init.T):
            edge_tuple = tuple(edge.tolist())

            if edge_tuple in adj_post_set:
                # Get index of the edge in adj_post
                idx = (adj_post.T == edge).all(dim=1).nonzero(as_tuple=True)[0]
                if len(idx) > 0:  # Edge exists in updated graph
                    edge_labels[i] = 1 if abs(edge_attr_post[idx[0],0]) >= threshold or abs(edge_attr_post[idx[0],1]) >= threshold  else 0

        return torch.tensor(edge_labels)



    def get_node_features(self, node_data_pre, node_data_post, gen_data_pre):
        '''
        extracts the unnormalized node features and labels from the raw data
        
        Input:
            node_data_pre:  node data read from matpower formatted matlab file of the initial state
            node_data_post: node data read from matpower formatted matlab files of the post cascading failure state
        Output:
            node_features:  torch.tensor of node features
            node_labels:    torch.tensor of node labels
        '''


        #one hot encoded bus types
        N_BUSES = len(node_data_pre[:,2])
        bus_type = torch.zeros([2000,4], dtype=torch.int32)
        bus_type2 = torch.zeros([2000,4], dtype=torch.int32)
        for i in range(N_BUSES):
            bus_type[i, int(node_data_pre[i,1]-1)] = 1
            bus_type2[i, int(node_data_post[i,1]-1)] = 1

        P1 = torch.tensor(node_data_pre[:,2]) #P of all buses at initial condition - Node feature
        Q1 = torch.tensor(node_data_pre[:,3]) #Q of all buses at initial condition - Node feature
        S1 = np.sqrt(P1**2+Q1**2).clone().detach()
        Vm = torch.tensor(node_data_pre[:,7]) #Voltage magnitude of all buses at initial condition - Node feature
        Va = torch.tensor(node_data_pre[:,8]) #Voltage angle of all buses at initial condition - Node feature
        baseKV = torch.tensor(node_data_pre[:,9]) #Base Voltage
        Vm = Vm*baseKV
        P2 = torch.tensor(node_data_post[:,2]) #P of all buses after step - used for calculation of Node labels
        Q2 = torch.tensor(node_data_post[:,3]) #Q of all buses after step - used of calculation of Node labels
        S2 = np.sqrt(P2**2+Q2**2).clone().detach()
        if self.data_type in ['AC', 'n-k']:
            Bs = torch.tensor(node_data_pre[:,5]) #Shunt susceptance
            Bs[bus_type[:,3]==1] = 0
        
        #one hot encoded node IDs
        node_ID = torch.eye(N_BUSES)
        
        #adjust features of inactive buses
        P1[bus_type[:,3]==1] = 0
        Q1[bus_type[:,3]==1] = 0
        S1[bus_type[:,3]==1] = 0
        Vm[bus_type[:,3]==1] = 0
        Va[bus_type[:,3]==1] = 0

        P2[bus_type2[:,3]==1] = 0
        Q2[bus_type2[:,3]==1] = 0
        S2[bus_type2[:,3]==1] = 0
        
        
        gen_features = self.get_gen_features(gen_data_pre, node_data_pre)
        gen_features[bus_type[:,3]==1,:] = 0

        #node Features for AC (ANGF_CE_Y) and n-k data
        if self.data_type in ['AC', 'n-k']:
            node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), Bs.reshape(-1,1), bus_type, gen_features, node_ID], dim=1)
            node_labels = torch.tensor(S1-S2)
        #Node features for Zhu data
        elif self.data_type in ['Zhu', 'Zhu_mat73', 'LSTM', 'Zhu_nobustype']:
            P_injection = (gen_features[:,0]-P1)
            Q_injection = (gen_features[:,1]-Q1)
            Vreal = Vm*torch.cos(np.deg2rad(Va))
            Vimag = Vm*torch.sin(np.deg2rad(Va))
            #ajust values to bus types according to Zhu paper
            if self.data_type in ['Zhu', 'zhu_mat73']:
                P_injection = P_injection*(bus_type[:,0]+bus_type[:,1])  #P only given for PQ and PV buses
                Q_injection = Q_injection*(bus_type[:,0])  #Q only given for PQ
                Vreal = Vreal*(bus_type[:,1]+bus_type[:,2])  #V only given for PV and slack bus
                Vimag = Vimag*(bus_type[:,1]+bus_type[:,2])


            Vm2 = torch.tensor(node_data_post[:,7]*node_data_post[:,9]) #P of all buses after step - used for calculation of Node labels
            Va2 = torch.tensor(node_data_post[:,8]) #Q of all buses after step - used of calculation of Node labels
            Vm2[bus_type2[:,3]==1] = 0
            Va2[bus_type2[:,3]==1] = 0
            V2real = Vm2*torch.cos(np.deg2rad(Va2))
            V2imag = Vm2*torch.sin(np.deg2rad(Va2))

            node_features = torch.cat([P_injection.unsqueeze(1), Q_injection.unsqueeze(1), Vreal.unsqueeze(1), Vimag.unsqueeze(1), bus_type], dim=1)
            node_labels = torch.cat([V2real.unsqueeze(1), V2imag.unsqueeze(1)], dim=1)     #S1-S2 is passed not to be used as node feature but for the graph labels

        #Node features for ANGF_Vcf data
        elif self.data_type == 'ANGF_Vcf':
            Vreal = Vm*torch.cos(np.deg2rad(Va))
            Vimag = Vm*torch.sin(np.deg2rad(Va))
            P_injection = (gen_features[:,0]-P1)
            Q_injection = (gen_features[:,1]-Q1)
            node_features = torch.cat([P_injection.reshape(-1,1), Q_injection.reshape(-1,1), Vreal.reshape(-1,1), Vimag.reshape(-1,1), bus_type, gen_features[:,2:]], dim=1)
            node_labels = torch.tensor(S1-S2)

        else:
            node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), gen_features], dim=1)
            node_labels = torch.tensor(S1-S2)
        graph_label = (S1-S2).unsqueeze(1).sum()
         
        return node_features, node_labels, graph_label
    
    def get_gen_features(self, gen_data_pre, node_data_pre):
        if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:
            gen_features = torch.zeros(2000, 9)
        else: gen_features = torch.zeros(2000,2)
        node_index = 0
        for i in range(len(gen_data_pre)):
            while gen_data_pre[i,0] != node_data_pre[node_index,0]: #get the node belonging to the generator
                node_index += 1
                if node_index >= 2000: node_index = 0
            if gen_data_pre[i,0] == node_data_pre[node_index,0]:

                if gen_data_pre[i,7] >0 and node_data_pre[node_index,1]!=4:    #if generator is active and bus is active
                    gen_features[node_index][:2] += torch.tensor(gen_data_pre[i,1:3])    #only adds p and q if the generator is active since ac-cfm does not update inactive buses
                    if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                        gen_features[node_index][6] = 1
                        if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                        else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                        gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                        if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                        else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])

                elif node_data_pre[node_index,1] != 4:   #if gen is inactive but bus is active
                    if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                        gen_features[node_index][6] = gen_features[node_index][6]   #if bus is active but generator isnt leave state as is since an active gen could be connected
                        #set lower limits and voltage set point only to inactive values if there are no existing values yet
                        if gen_features[node_index][3] == 0: gen_features[node_index][3] = gen_data_pre[i,4]    #Pmin
                        if gen_features[node_index][4] == 0: gen_features[node_index][4] = gen_data_pre[i,5]    #voltage set point
                        if gen_features[node_index][8] == 0: gen_features[node_index][8] = gen_data_pre[i,9]    #Qmin  

                else:   #this case is only entered if bus is inactive then all gens should also be counted as inactive 
                    gen_features[node_index][:2] = 0
                    if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                        if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                        else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                        gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                        gen_features[node_index][6] = 0     
                        if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                        else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])
                        
                if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #features that are treated equally for active and inactive busses and generatos
                    gen_features[node_index][2] += torch.tensor(gen_data_pre[i,3])    
                    gen_features[node_index][5] += torch.tensor(gen_data_pre[i,6])
                    gen_features[node_index][7] += torch.tensor(gen_data_pre[i,8])
                    

                
        if self.data_type in ['AC', 'n-k', 'ANGF_Vcf']:
            gen_features = torch.cat([gen_features[:,:6], gen_features[:,7:], gen_features[:,6].reshape(-1,1)], dim=1)

        
        return gen_features

    def decode_damage(self, dmgs, step, node_IDs, edge_IDs):
        """node_IDs must be passed as a list of node IDs
        edge_IDs must be passed as a list of edge IDs
        dmgs must be passed as a list of damages with the format [[step, edge_ID], ...]"""
        decoded_damages = []

        for i in range(len(dmgs)):
            if dmgs[i,0] == step:
                dmg = dmgs[i]
                for busID in range(len(node_IDs)):
                    if node_IDs[busID] == edge_IDs[dmg[1]-1, 0]:  busID_a = busID
                for busID in range(len(node_IDs)):
                    if node_IDs[busID] == edge_IDs[dmg[1]-1, 1]:  busID_b = busID
                decoded_damages.append([busID_a, busID_b])

        return decoded_damages
            


    def get_edge_attrY(self, edge_data, decoded_damages):
        "decoded_damages is encoded as [[bus_from, bus_to]], with python indices (0-1999)"
        #Deactivate damaged lines
        if decoded_damages != []:
            for i in range(len(decoded_damages)):
                edge_data[decoded_damages[i][0],decoded_damages[i][1]] = 0
                edge_data[decoded_damages[i][1],decoded_damages[i][0]] = 0
        # Threshold value
        threshold = 1e-8

        # Step 1: Get the indices of entries that satisfy the condition > 1e-8
        if len(edge_data) == 1: 
            edge_data = torch.complex(torch.tensor(edge_data[0]['real']), torch.tensor(edge_data[0]['imag']))
            mask = np.abs(edge_data) > threshold
        else:                   
            mask = np.abs(edge_data) > threshold
        edge_index = torch.tensor(mask).nonzero().t()



        # Step 2: Extract the corresponding edge attributes (weights)
        
        edge_attr = torch.cat([torch.tensor(edge_data[edge_index[0], edge_index[1]].real).unsqueeze(1), torch.tensor(edge_data[edge_index[0], edge_index[1]].imag).unsqueeze(1)], dim=1)
        return edge_index, edge_attr
    
    def get_edge_attrY_Zhumat73(self, edge_data, decoded_damages):
        #Deactivate damaged lines
        for i in range(len(decoded_damages)):
            edge_data[decoded_damages[i][0],decoded_damages[i][1]] = 0
            edge_data[decoded_damages[i][1],decoded_damages[i][0]] = 0
        # Threshold value
        threshold = 1e-8

        # Step 1: Get the indices of entries that satisfy the condition > 1e-8
        mask = abs(edge_data) > threshold
        edge_index = torch.tensor(mask).nonzero().t()

        # Step 2: Extract the corresponding edge attributes (weights)
        edge_attr = torch.cat([torch.tensor(edge_data[edge_index[0], edge_index[1]].real).unsqueeze(1), torch.tensor(edge_data[edge_index[0], edge_index[1]].imag).unsqueeze(1)], dim=1) 
        #edge_attr = torch.cat([torch.complex(torch.tensor(edge_data[edge_index[0], edge_index[1]][0]), torch.tensor(edge_data[edge_index[0], edge_index[1]][1]))], dim=1)
        return edge_index, edge_attr


    def get_edge_features(self, edge_data, damages, node_data_pre, scenario, i, n_minus_k):

        rating = edge_data[:,5] #long term rating (MVA) - edge feature
        status = edge_data[:,10]  #1 if line is working 0 if line is not - edge feature
        resistance = edge_data[:,2]
        reactance = edge_data[:,3] 
        #power flows
        pf1 = edge_data[:,13]
        qf1 = edge_data[:,14]

        pf2 = edge_data[:,15]
        qf2 = edge_data[:,16]

        Gs = torch.tensor(node_data_pre[:,4]) #Shunt conductance
        Bs = torch.tensor(node_data_pre[:,5]) #Shunt susceptance
                
        #Adjacency Matrix
        bus_id = node_data_pre[:,0] #list of bus ids in order
        bus_from = edge_data[:,0]   
        bus_to = edge_data[:,1] 

        #initial damages
        init_dmg = torch.zeros(len(status)) #edge feature that is 0 except if the line was an initial damage during that step
        #set initially damaged lines to 1
        for step in range(len(damages[scenario])):
            if n_minus_k:
                if damages[scenario][step,0] <= i:
                    init_dmg[damages[scenario][step,1]-1] = 1 
            else:       
                if damages[scenario][step,0] == i:
                    init_dmg[damages[scenario][step,1]-1] = 1 
                    j = 0
                    while bus_from[damages[scenario][step,1]-1] == bus_from[damages[scenario][step,1]-1-j] and bus_to[damages[scenario][step,1]-1] == bus_to[damages[scenario][step,1]-1-j]:
                        init_dmg[damages[scenario][step,1]-1-j] = 1 
                        j = j+1
                    j = 0
                    while bus_from[damages[scenario][step,1]-1] == bus_from[damages[scenario][step,1]-1+j] and bus_to[damages[scenario][step,1]-1] == bus_to[damages[scenario][step,1]-1+j]:
                        init_dmg[damages[scenario][step,1]-1+j] = 1 
                        j = j+1

        
        #Features
        adj_from = []   #adjacency matrix from/to -> no edges appearing twice
        adj_to = []
        rating_feature = [] #new list because orig data contains multiple lines along the same edge which are combined here
        resistance_feature = []
        reactance_feature = []
        init_dmg_feature = [] #always zero except if the line was an initial damage during this step -> then 1
        
        pf_feature = []
        qf_feature = []
        #Add edges and the respective features, edges are always added in both directions, so that the pf can be directional, the other features
        #   are added to both directions
        for j in range(len(bus_from)):
            id_from = int(np.where(bus_id==bus_from[j])[0]) #bus_id where line starts
            id_to = int(np.where(bus_id==bus_to[j])[0])     #bus_id where line ends
            
            #if edge already exists recalculate (add) the features and dont add a new edge
            exists=False
            if (adj_from.count(id_from) > 0):   #check if bus from exists
                for k in range(len(adj_from)):  #check all appeareances of bus from in adj_from
                    if adj_from[k] == id_from and adj_to[k] == id_to: #if bus from and bus to are at the same entry update their edge features
                        exists = True                       #mark as edge exists
                        """if status[j] == 0:      #remove existing edge if status is inactive
                            adj_from.remove(k)
                            adj_to.remove(k)"""
                        if status[j]==1:
                            rating_feature[k] += rating[j]      #add the capacities (ratings)
                            resistance_feature[k], reactance_feature[k] =self.calc_total_resistance_reactance(resistance_feature[k], resistance[j], reactance_feature[k], reactance[j])
                            pf_feature[k] += pf1[j]         #add PF
                            qf_feature[k] += qf1[j]         #add PF
 
                        if init_dmg_feature[k] != 1:
                            init_dmg_feature[k] = init_dmg[j]

            if (adj_to.count(id_from)>0):       #check other way
                for k in range(len(adj_to)):
                    if adj_to[k] == id_from and adj_from[k] == id_to:
                        exists = True

                        """if status[j] == 0:
                            adj_to.remove(k)
                            adj_from.remove(k)"""
                        if status[j] == 1:
                            rating_feature[k] += rating[j]      #add the capacities (ratings)
                            resistance_feature[k], reactance_feature[k] = self.calc_total_resistance_reactance(resistance_feature[k],resistance[j],reactance_feature[k], reactance[j])
                            pf_feature[k] += qf2[j]
                            qf_feature[k] += qf2[j]

                        if init_dmg_feature[k] != 1:
                            init_dmg_feature[k] = init_dmg[j]
                        
            if exists: 
                continue
            #if edge does not exist yet add it in both directions
            elif status[j]==1 and init_dmg[j] == 0:
                
                #First direction
                
                adj_from.append(id_from)
                adj_to.append(id_to)
    
                #status_feature.append(status[j])
    
                init_dmg_feature.append(init_dmg[j])
                if status[j] !=0:
                    pf_feature.append(pf1[j])   #pf in first directiong
                    qf_feature.append(qf1[j])   #qf in first directiong
                    pf_feature.append(pf2[j])   #pf in opposite direction
                    qf_feature.append(qf2[j])   #qf in opposite direction
                    resistance_feature.append(resistance[j])
                    reactance_feature.append(reactance[j])
                    rating_feature.append(rating[j])
                else:
                    pf_feature.append(0)   #if line inactive set power flows to 0 for both directions
                    qf_feature.append(0)   
                    pf_feature.append(0)   
                    qf_feature.append(0)   
                    resistance_feature.append(1)
                    reactance_feature.append(1)
                    rating_feature.append(0)
                #Opposite direction
                adj_from.append(id_to)
                adj_to.append(id_from)
                rating_feature.append(rating[j])
                #status_feature.append(status[j])
                resistance_feature.append(resistance[j])
                reactance_feature.append(reactance[j])
                init_dmg_feature.append(init_dmg[j])
            


        adj = torch.tensor([adj_from,adj_to])

        if self.edge_attr == 'Y':
            impedance = torch.tensor([resistance_feature, reactance_feature])
            impedance = torch.transpose(impedance, 0, 1).contiguous()   #(5154, 2)
            impedance_complex = torch.view_as_complex(impedance)        #(5154)
            admittance_complex = torch.reciprocal(impedance_complex)    #(5154)
            
            #edge_attr = - torch.view_as_real(admittance_complex)
            edge_attr = -admittance_complex

            Y = torch.zeros((2000,2000), dtype=torch.cfloat)
            for idx, edge in enumerate(adj.t().tolist()):                
                source, target = edge
                Y[source, target] = - admittance_complex[idx]
            admittance_sum = torch.sum(Y, dim=0) #(2000), contains (y12 + y13, y12 + y23, ...)
            
            self_admittance = torch.complex(Gs, Bs) + admittance_sum
            #self_admittance = self_admittance + torch.view_as_real(admittance_sum) #DO POPRAWKI
            
            edge_attr = torch.cat([edge_attr, self_admittance], dim=1)
            #edge_attr = torch.transpose(edge_attr, 0, 1)
            #edge_attr = np.sqrt(edge_attr[0,:]**2+edge_attr[1,:]**2).clone().detach()

            self_connections = torch.stack([torch.arange(2000), torch.arange(2000)], dim=0)
            adj = torch.cat([adj, self_connections], dim=1)
        else:
            edge_attr = torch.tensor([rating_feature, pf_feature, qf_feature, resistance_feature, reactance_feature, init_dmg_feature])
            edge_attr = torch.transpose(edge_attr,0,1)

        
        return adj, edge_attr

      
    def find_nans(status, feature, scenario, i):
        #Check for NaNs
        problems = []
        
        for j in np.where(np.isnan(feature))[0]:
            if status[j]==1: 
                problems.append([scenario,i,j])
        return problems

            
    #TO
    def get_scenario_of_file(self,name):
        """
        Input:
        name        name of the processed data file
        
        Returns:
        scenario    index of the scenario of which the datafile stems
        """
        """if name.startswith('./processed'):
            name=name[17:]
        else:
            name=name[26:]
        i=1"""
        i = 1
        while not name[i].isnumeric():
            i+=1
        j = 1
        while name[j+i].isnumeric():
            j+=1
        scenario=int(name[i:i+j])
        return scenario
    
    #TO
    def get_scenario_step_of_file(self,name):  
        """
        

        Parameters
        ----------
        name : string
                name of the processed data file

        Returns
        -------
        scenario : int
            Scenario of which the file stems
        step : int
            Step in that scenario

        """
        name=name[5:]
        i=0
        while name[i].isnumeric():
            i+=1
        scenario=int(name[0:i])
        j=i+1
        while name[j].isnumeric():
            j+=1
        step=int(name[i+1:j])
        return scenario,step
                
    
    def get_initial_damages(self):
        '''
        returns the sorted initial damages of each scenario in damages [N_scenarios][step,line_id]
        where step is reassigned so that it starts with 0 and increments by 1 except if two lines 
        were destroyed in the same step
        '''
        
        #load scenario file which stores the initial damages
        f = open('raw/Hurricane_Ike_gamma8.3e-5_scenarios.txt','r')
        lines= f.readlines()
        damages = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace("[", '')
            lines[i] = lines[i].replace(']', '')
            lines[i] = lines[i].replace('(', '')
            lines[i] = lines[i].replace(')', '')
            lines[i] = lines[i].replace('"', '')
            lines[i] = lines[i].replace(',', '')
            line = np.array(list(map(int, lines[i].split())))
            scenario_dmgs=np.reshape(line,(-1,2))
            scenario_dmgs=scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]
            #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
            index = 0
            for j in range(0,len(scenario_dmgs)-1):
                increment = 0
                if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
                    increment = 1
                scenario_dmgs[j,0] = index
                index += increment
            scenario_dmgs[-1,0] = index

            damages.append(scenario_dmgs)
        return damages
    

    def get_initial_damages_dc(self, scenario_dmgs):
        '''

        '''

        scenario_dmgs = np.array(scenario_dmgs).astype(int)
        scenario_dmgs[:,[0,1]] = scenario_dmgs[:,[1,0]]
        scenario_dmgs = scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]        

        #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
        index = 0
        for j in range(0,len(scenario_dmgs)-1):
            increment = 0
            if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
                increment = 1
            scenario_dmgs[j,0] = index
            index += increment
        scenario_dmgs[-1,0] = index

        return scenario_dmgs

            

    def download(self):
        pass
    
    def calc_total_resistance_reactance(self, r1, r2, x1, x2):
        #calculates the total resistance of the existing edge and the line to be added
        a = (r1**2-x1**2)*(r2**2-x2**2)
        G = (r1*r2**2-r1*x2**2+r2*r1**2-r2*x1**2)/a
        B = (x1*x2**2-x1*r2**2-x2*r1**2+x2*x1**2)/a
        r_new = G/(G**2+B**2)
        x_new = B/(G**2 + B**2)
        return r_new, x_new
    
    
    def len(self):
        return len(self.processed_file_names)
    
    
    def __getitem__(self,idx):
        #scenario=int(self.data_list[idx,0])
        #step=int(self.data_list[idx,1])
        #data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'f'_{step}.pt'))
        data = torch.load(os.path.join(self.processed_dir, self.data_list[idx]))
        data.x = data.x[:, :4]

        #if self.embedding != None:
            #embedding = torch.cat([self.embedding]*int(len(data.x)/2000))
            #print(f'Embedding shape: {embedding.shape}')
            #print(f'self.embedding shape: {self.embedding.shape}')
            #data.x = torch.cat([data.x.to('cpu'), self.embedding.to('cpu')], dim=1)
        return data
    
    def load_mat_file(self, file_path):
        try:
        # Attempt to load using scipy (works for MATLAB files below v7.3)
            #data = scipy.io.loadmat(file_path)
            #filetype = 'Zhu'
            return scipy.io.loadmat(file_path), 'Zhu'   #data, filetype

        except NotImplementedError:
            f = h5py.File(file_path, 'r')
            data = f
            filetype = 'Zhu_mat73'

            return data, filetype
        
    def process_n_minus_k(self):
        """
        Loads the raw matlab data, converts it to torch
        tensors which are then saved. Nees to be manually normalized after processing using normalize.py
        """
        #INIT
        #load scenario file which stores the initial damages
        damages = self.get_initial_damages()
        #load initial network data
        init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
        #Initial data
        node_data_pre = init_data['ans'][0,0][2]    #ans is correct bcs its pwsdata
        gen_data_pre = init_data['ans'][0,0][3]
        edge_data = init_data['ans'][0,0][4]
        below_threshold_count = 0

        #PROCESSING
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
                continue
            scenario = self.get_scenario_of_file(raw_path)
            file=scipy.io.loadmat(raw_path)  #loads a full scenario 

            accumulated_ls_tot = 0.  #used for step selection according to ls_threshold and N_below_threshold
            remaining_load = 1.      #necessary to correctly scale ls_tot since it refers to load shed relative to initial load at each step (not init load of scenario)

            #loop through steps of scenario each step will be one processed data file
            for i in range(min(len(file['clusterresult'][0,:]), 10)):
                accumulated_ls_tot += file['clusterresult'][0,i][21] * remaining_load
                remaining_load -= file['clusterresult'][0,i][21] * remaining_load

                #skip if total loadshed of timestep is below threshold and the amount of low loadshed instances is reached
                if self.data_type == 'LSTM' or accumulated_ls_tot>self.ls_threshold or below_threshold_count<self.N_below_threshold:
                    if below_threshold_count<self.N_below_threshold and accumulated_ls_tot<self.ls_threshold:
                        below_threshold_count += 1

                    if np.isnan(file['clusterresult'][0,i][21]):    #This refers to matlab column ls_total -> if this is NaN the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                        print('Skipping because ls_tot==NaN', file, i)
                        continue
                    node_data_post = file['clusterresult'][0,i][2]   #node_data after step i for node_label_calculation

                    node_feature, node_labels = self.get_node_features(node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data                    

                    adj, edge_attr = self.get_edge_features(edge_data, damages, node_data_pre, scenario, i, n_minus_k=True)



                    graph_label = node_labels.sum()
                    
                    data = Data(x=node_feature.float(), edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1), node_labels=node_labels, y=graph_label) 
                    torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
      

def collate_fn(batch):
    sequences = [item.x for item in batch]
    targets = torch.tensor([item.y for item in batch], dtype=torch.float32)
    targets_class = torch.tensor([item.y_class for item in batch], dtype=torch.long)    #torch.stack([item.y_class for item in batch])
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)

    return padded_sequences, targets, lengths, targets_class

def collate_fn_fixed_length(batch, max_length):
    # Extract sequences and target variables
    sequences = [item.x for item in batch]
    targets = [item.y_seq for item in batch]
    targets_class = [item.y_seq_class for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # Determine the maximum sequence length
    max_len = max_length
    # Manually pad sequences on the left
    padded_sequences = torch.zeros(len(sequences), max_len, sequences[0].size(-1))  # Initialize with zeros
    padded_targets = torch.zeros(len(sequences), max_len)  # Initialize with zeros
    padded_targets_class = torch.zeros(len(sequences), max_len)  # Initialize with zeros
    for i, seq in enumerate(sequences):
        padded_sequences[i, -lengths[i]:] = seq  # Place the sequence at the end, leaving padding at the start
        padded_targets[i, -lengths[i]:] =torch.tensor(targets[i])  # Place the sequence at the end, leaving padding at the start
        padded_targets_class[i, -lengths[i]:] = torch.tensor(targets_class[i])
    lengths[:] = max_len


    return padded_sequences, padded_targets, lengths, padded_targets_class

    

def create_datasets(root ,cfg, pre_transform=None, num_samples=None, stormsplit=0, embedding=None, data_type = 'AC', edge_attr='multi'):
    """
    Helper function which loads the dataset and splits it into a training and a
    testing set.
    Input:
        root (str) : the root folder for the dataset
    Return:
        trainset : the training set
        testset : the testset
        data_list : the data_list
    """
    print('Creating Datasets...')
    t1 = time.time()
    print(t1, flush=True)
    dataset = HurricaneDataset(root=root,use_supernode=cfg["supernode"], pre_transform=pre_transform,N_Scenarios=cfg["n_scenarios"], stormsplit=stormsplit, 
                               embedding=embedding, data_type=data_type, edge_attr=edge_attr, ls_threshold=cfg['ls_threshold'], N_below_threshold=cfg['N_below_threshold'])
    #data_list = dataset.data_list

    if num_samples is None:
        len_dataset=len(dataset)
    else:
        print("Error: create_datasets can not accept num_samples as input yet")
    print(f'Len Dataset: {len_dataset}')
    #Get last train sample if stormsplit
    if stormsplit != 0:
        print('NO STORMSPLIT AT THE MOMENT')
    #    for i in range(len(data_list)):
    #        if str(data_list[i,0]).startswith(str(stormsplit)):
    #            last_train_sample=i
    #            break

    #Get last train sample if no stormsplit
    else:   
        trainsize = cfg["train_size"]
        last_train_sample = int(len_dataset*trainsize)
        """if trainsize <1:
            while data_list[last_train_sample-1,0]==data_list[last_train_sample,0]:
                last_train_sample+=1
            #testset = Subset(dataset, range(last_train_sample, len_dataset))
        #else: testset= Subset(dataset,range(len_dataset,len_dataset))"""
    
    #trainset = Subset(dataset, range(0, last_train_sample))
    #testset = Subset(dataset, range(last_train_sample, len_dataset))

    trainset, testset = random_split(dataset, [last_train_sample, len_dataset-last_train_sample])
    
    t2 = time.time()
    print(f'Creating datasets took {(t2-t1)/60} mins', flush=True)

    return trainset, testset#, data_list 

def lstm_get_max_seq_length(trainset, testset):
    """
    Helper function which calculates the maximum sequence length of the
    training and test set.
    Input:
        trainset : the training set
        testset : the testing set
    Return:
        max_length : the maximum sequence length
    """
    print('Calculating maximum sequence length...')
    t1 = time.time()
    max_length = 0
    for i in range(len(trainset)):
        max_length = max(max_length, trainset[i].x.shape[0])
    for i in range(len(testset)):
        max_length = max(max_length, testset[i].x.shape[0])
    t2 = time.time()
    print(f'Calculating maximum sequence length took {(t2-t1)/60} mins')
    return max_length

def create_loaders(cfg, trainset, testset, pre_compute_mean=False, Node2Vec=False, data_type='AC', num_workers=0, pin_memory=False, task='GraphReg'): 
    """
    Helper function which creates the dataloaders and
    pre-computes the means of the testset labels for more
    efficient R2 computation.
    Input:
        cfg (dict) : the configuration dictionary containing
            parameters for the loaders
        trainset : the training dataset
        testset : the testing dataset
        pre_compute_mean (bool) : descides whether mean is
            computed or not
        Node2Vec (bool) : if True the trainloader is created with batchsize one for usage of Node2Vec
    Return:
        trainloader : the training set loader
        testloader : the testing set loader
    """
    print('Creating Dataloaders...')
    t1 = time.time()
    max_length = -1
    if Node2Vec:
        trainloader = DataLoader(trainset, batch_size=1, shuffle=cfg["train_set::shuffle"]        
        )
        """elif data_type == 'LSTM':
            trainloader = DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], collate_fn=collate_lstm)
            testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"], collate_fn=collate_lstm)"""
    elif 'LDTSF' in data_type:
        if 'typeII' in task:
            max_length = lstm_get_max_seq_length(trainset, testset)
            print(max_length)
            collate = partial(collate_fn_fixed_length, max_length=max_length)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
            testloader = torch.utils.data.DataLoader(testset, batch_size=cfg["test_set::batchsize"], collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
            testloader = torch.utils.data.DataLoader(testset, batch_size=cfg["test_set::batchsize"], collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    else:
        trainloader = DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"], num_workers=num_workers, pin_memory=pin_memory)

    """if pre_compute_mean:
        mean_labels = 0.
        for batch in testloader:   
            mean_labels += batch.y.sum().item()
        mean_labels /= len(testloader) 
        testloader.mean_labels = mean_labels"""

    print(f'Creating dataloaders took {(time.time()-t1)/60} mins')
    return trainloader, testloader, max_length




def calc_mask_probs(dataloader, cfg):  
    """
    Calculates the masking probabilities based on the variance of the node
    If masking is turned off returns an array of ones (equivalent to no masking)
    
    Parameters
    ----------
    dataloader : the dataloader for the dataset for which the masking probabilities should be calculated

    Returns
    -------
    node_label_probs : float array
        the masking probabilities

    """

    if cfg['use_masking'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
        if isfile('node_label_vars.pt'):
            print('Using existing Node Label Variances for masking')
            mask_probs = torch.load('node_label_vars.pt')
        else:
            print('No node label variance file found\nCalculating Node Variances for Masking')
            node_label_vars=np.zeros(2000)
            for i, batch in enumerate(dataloader):
                if i==0:
                    labels=batch.node_labels.clone()
                else:
                    labels=torch.cat((labels,batch.node_labels))
                
            labels=labels.reshape( int(len(labels)/2000),2000)

            for i in range(2000):
                node_label_vars[i] = labels[:,i].var()
            #scale vars
            print(f'MAX {node_label_vars.argmax()}')
            node_label_probs = torch.tensor(node_label_vars/node_label_vars.max())
            mask_probs = node_label_probs
            torch.save(mask_probs, 'node_label_vars.pt')
    else:
        #Masks are set to one in case it is wrongly used somewhere (when set to 1 masking results in multiplication with 1)
        mask_probs = torch.zeros(2000)+1

    return mask_probs



def mask_probs_add_bias(mask_probs, bias):
    """
    mask_probs  : float array (1D)
        the masking probabilities of the nodes
    bias        : float
         the bias to be added to the masking probabilities
     Returns
     mask_probs_rescaled : float array (1D)
         the masking probabilities with added bias
    """
    
    mask_probs_rescaled = mask_probs.clone() + bias
    for i in range(len(mask_probs)):
        if mask_probs_rescaled[i] > 1.0: mask_probs_rescaled[i] = 1
    return mask_probs_rescaled

def get_attribute_sizes(cfg, trainset):
    """
    Used to get the sizes of node_features, edge_features and targets depending on data and task
    """
    #Get number of node features
    if cfg['data'] in ['LSTM']:
        num_features = trainset.__getitem__(0)[0].x.shape[1]
    else:
        num_features = trainset.__getitem__(0).x.shape[1]

    #Get number of edge features
    if 'LDTSF' not in cfg['data']: 
        if cfg['data'] in ['LSTM']:
            if trainset.__getitem__(0)[0].edge_attr.dim() == 1:
                if cfg['edge_attr'] == 'multi':     print('WARNING: CONFIG SET TO MULTIPLE FEATURES BUT DATA CONTAINS ONLY 1!')
                num_edge_features = 1
            else:
                num_edge_features = trainset.__getitem__(0)[0].edge_attr.shape[1]
        else:
            if trainset.__getitem__(0).edge_attr.dim() == 1:
                if cfg['edge_attr'] == 'multi':     print('WARNING: CONFIG SET TO MULTIPLE FEATURES BUT DATA CONTAINS ONLY 1!')
                num_edge_features = 1
            else:
                num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
    else:
        num_edge_features = 0

    #Get number of targets
    if cfg['data'] in ['Zhu', 'Zhu_nobustype']:                      
        if cfg['task'] in ['NodeReg']:  num_targets = 2 #Zhu has Real and imaginary part as nodelabels
        else:                           num_targets = 1
    elif cfg['task'] in ['GraphReg', 'NodeReg']:    num_targets = 1 #All other data has power outage as node label except for the classification tasks
    elif cfg['task'] == 'typeIIClass':              num_targets = 2
    else:                                           num_targets = 4  

    return num_features, num_edge_features, num_targets



def save_node2vec(embedding, labels, data_list):
    """
    Saves the data node2vec embeddings
    deprecated??
    embedding   :   float array (2D)
        Node2Vec embedding
    labels  : float array (1D)
        the labels
    data_list   : int array (2D)
        the data list
        
    Returns
    """
    print(embedding.shape)
    print(len(embedding))
    embedding = embedding.reshape(int(len(embedding)/2000), 2000, embedding.shape[1])
    labels = labels.reshape(int(len(labels)/2000),2000)
    embedding = embedding.half()
    if not os.path.exists('node2vec/'):
        os.makedirs('node2vec/')
    for i in range(len(embedding)):
        x = embedding[i].data.half()
        y = labels[i].data.half()
        
        data=Data(x=x ,y=y)
        torch.save(data, f'node2vec/data_{int(data_list[i,0])}_{int(data_list[i,1])}.pt')
    

