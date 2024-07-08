"""
Author : Tobias Ohlinger

File setting up training and test data as well as neural network models
"""
import os
import numpy as np
import scipy.io
import time
import torch

from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch_geometric.utils import to_undirected
from torch.nn.utils.rnn import pad_sequence






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

    
    def __init__(self, root,use_supernode, transform=None, pre_transform=None, pre_filter=None, N_Scenarios=100, stormsplit=0, embedding=None, device=None, data_type='AC', edge_attr='multi', ls_threshold = .09, N_below_threshold=1):
        self.use_supernode=use_supernode
        self.embedding = embedding
        self.device = device
        self.data_type = data_type
        self.edge_attr = edge_attr
        self.ls_threshold = ls_threshold
        self.N_below_threshold = N_below_threshold
        super().__init__(root, transform, pre_transform, pre_filter)
        self.stormsplit = stormsplit
        self.data_list=self.get_data_list(N_Scenarios)  #list containing all instances in order
        #print(self.data_list)
        
        
    
    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = os.listdir(self.root + "/processed")
        return [f for f in files if f.startswith("data")]
        
    
    def get_data_list(self,N_scenarios):
        #test_id is the id given to the storm when compiling the dataset of all storms (i.e. the first digit of the scenario (f.e. Claudette=1)) and is used to relate the data files to the storms
        #To use the percentage based train test split stormsplit should be set to 0
        #N_scenario must be last Scenario that appears in raw (if scenario 1,2 and 100 are used N_scenarios must be 100)
        data_list=np.zeros((len(self.processed_file_names),2))
        idx=0  
        test_idx=0  
        if self.stormsplit == 0:                      
            for i in range(N_scenarios):
                first=True
                for file in self.processed_file_names:
                    if file.startswith(f'data_{i+1}_'):
    
                        _,step=self.get_scenario_step_of_file(file)
                        if first:
                            first=False
                            scenario_list=[step]
                        else:
                            scenario_list.append(step)
                if not first:
    
                    scenario_list=np.sort(np.array(scenario_list))
    
                    data_list[idx:idx+len(scenario_list),1]=scenario_list
                    data_list[idx:idx+len(scenario_list),0]=i+1
                    idx+=len(scenario_list)
             
                
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
        if self.data_type == 'AC' or self.data_type == 'LSTM':
            self.process_ac()
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
        #load initial network data
        init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
        num_nodes = len(init_data['ans'][0,0][2])
        below_threshold_count = 0


        #PROCESSING
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
                continue
            scenario = self.get_scenario_of_file(raw_path)
            file=scipy.io.loadmat(raw_path)  #loads a full scenario 

            #loop through steps of scenario each step will be one processed data file
            for i in range(len(file['clusterresult'][0,:])):
                #skip if total loadshed of timestep is below threshold and the amount of low loadshed instances is reached
                if self.data_type == 'LSTM' or file['clusterresult'][0,i][21]>self.ls_threshold or below_threshold_count<self.N_below_threshold:
                    if below_threshold_count<self.N_below_threshold and file['clusterresult'][0,i][21]<self.ls_threshold:
                        below_threshold_count += 1
                    #Node data
                    if i == 0:  #in first iteration load original pwsdata as initial data  
                        node_data_pre = init_data['ans'][0,0][2]    #ans is correct bcs its pwsdata
                        gen_data_pre = init_data['ans'][0,0][3]
                        edge_data = init_data['ans'][0,0][4]
                    else:
                        node_data_pre = file['clusterresult'][0,i-1][2]   #node_data of initial condition of step i
                        gen_data_pre = file['clusterresult'][0,i-1][3]
                        edge_data = file['clusterresult'][0,i-1][4] #edge data of initial condition of step i
                    if np.isnan(file['clusterresult'][0,i][21]):    #This refers to matlab column ls_total -> if this is NaN the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                        print('Skipping', file, i)
                        continue
                    node_data_post = file['clusterresult'][0,i][2]   #node_data after step i for node_label_calculation
                    

                    node_feature, node_labels = self.get_node_features(node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data                    
                    adj, edge_attr = self.get_edge_features(edge_data, damages, node_data_pre, scenario, i, n_minus_k=False)
                    
                    graph_label = node_labels.sum()
                    
                    if self.data_type == 'LSTM':
                        if i == 0:
                            node_feature_seq = node_feature
                            edge_attr_seq = torch.transpose(edge_attr, 0, 1)
                            adj_seq = adj
                            graph_label_seq =graph_label
                        else:
                            node_feature_seq = torch.cat((node_feature_seq, node_feature))
                            edge_attr_seq = torch.cat((edge_attr_seq, torch.transpose(edge_attr, 0, 1)), dim=0)
                            adj_seq = torch.cat((adj_seq, adj+i*num_nodes), dim=1)
                            graph_label_seq += graph_label
                    
                                        
                    """if self.data_type == 'LSTM':
                        if i ==0:   data_seq = [data]
                        else:       data_seq.append(data)"""
                    #save unscaled data (non LSTM)
                    if self.data_type == 'AC':
                        data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr, node_labels=node_labels, y=graph_label) 
                        torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
            if self.data_type == 'LSTM':
                data = Data(x=node_feature_seq.float(), edge_index=adj_seq, edge_attr=edge_attr_seq, node_labels=node_labels, y=graph_label_seq) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
                #print('Batch in Save:\n', Batch.from_data_list(data_seq))
                #torch.save(Batch.from_data_list(data_seq), os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
            #np.save('problems',np.array(problems))



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
        num_nodes = len(init_data['ans'][0,0][2])
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
             
        P1 = torch.tensor(node_data_pre[:,2]) #P of all buses at initial condition - Node feature
        Q1 = torch.tensor(node_data_pre[:,3]) #Q of all buses at initial condition - Node feature
        S1 = torch.tensor(np.sqrt(P1**2+Q1**2))
        Vm = torch.tensor(node_data_pre[:,7]) #Voltage magnitude of all buses at initial condition - Node feature
        #Va = torch.tensor(node_data_pre[:,8]) #Voltage angle of all buses at initial condition - Node feature
        #Va = (Va-Va[0])%360
        if self.data_type in ['AC', 'n-k']:
            Bs = torch.tensor(node_data_pre[:,5]) #Shunt susceptance
            baseKV = torch.tensor(node_data_pre[:,9]) #Base Voltage
            Vm = Vm*baseKV
        
        P2 = torch.tensor(node_data_post[:,2]) #P of all buses after step - used for calculation of Node labels
        Q2 = torch.tensor(node_data_post[:,3]) #Q of all buses after step - used of calculation of Node labels
        S2 = torch.tensor(np.sqrt(P2**2+Q2**2))
        
        N_BUSES = len(node_data_pre[:,2])
 
        if self.data_type in ['AC', 'n-k']:
            #one hot encoded bus types
            bus_type = torch.zeros([2000,4], dtype=torch.int32)
            for i in range(N_BUSES):
                bus_type[i, int(node_data_pre[i,1]-1)] = 1
        
            #one hot encoded node IDs
            node_ID = torch.eye(N_BUSES)
        
        #adjust features of inactive buses
        P1[bus_type[:,3]==1] = 0
        Q1[bus_type[:,3]==1] = 0
        S1[bus_type[:,3]==1] = 0
        Vm[bus_type[:,3]==1] = 0
        #Va[bus_type[:,3]==1] = 0
        Bs[bus_type[:,3]==1] = 0
        
        gen_features = self.get_gen_features(gen_data_pre, node_data_pre)

            
        if self.data_type in ['AC', 'n-k']:
            node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), Bs.reshape(-1,1), bus_type, gen_features, node_ID], dim=1)
        else:
            node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), gen_features], dim=1)
        node_labels = torch.tensor(S1-S2)
         
        return node_features, node_labels
    
    def get_gen_features(self, gen_data_pre, node_data_pre):
        if self.data_type in ['AC', 'n-k']:
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
                    if self.data_type in ['AC', 'n-k']:  #Features not added for TimeSeries
                        gen_features[node_index][6] = 1
                        if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                        else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                        gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                        if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                        else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])

                elif node_data_pre[node_index,1] != 4:   #if gen is inactive but bus is active
                    if self.data_type in ['AC', 'n-k']:  #Features not added for TimeSeries
                        gen_features[node_index][6] = gen_features[node_index][6]   #if bus is active but generator isnt leave state as is since an active gen could be connected
                        #set lower limits and voltage set point only to inactive values if there are no existing values yet
                        if gen_features[node_index][3] == 0: gen_features[node_index][3] = gen_data_pre[i,4]    #Pmin
                        if gen_features[node_index][4] == 0: gen_features[node_index][4] = gen_data_pre[i,5]    #voltage set point
                        if gen_features[node_index][8] == 0: gen_features[node_index][8] = gen_data_pre[i,9]    #Qmin  

                else:   #this case is only entered if bus is inactive then all gens should also be counted as inactive 
                    gen_features[node_index][:2] = 0
                    if self.data_type in ['AC', 'n-k']:  #Features not added for TimeSeries
                        if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                        else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                        gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                        gen_features[node_index][6] = 0     
                        if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                        else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])
                        
                if self.data_type in ['AC', 'n-k']:  #features that are treated equally for active and inactive busses and generatos
                    gen_features[node_index][2] += torch.tensor(gen_data_pre[i,3])    
                    gen_features[node_index][5] += torch.tensor(gen_data_pre[i,6])
                    gen_features[node_index][7] += torch.tensor(gen_data_pre[i,8])
                    

                
        if self.data_type in ['AC', 'n-k']:
            gen_features = torch.cat([gen_features[:,:6], gen_features[:,7:], gen_features[:,6].reshape(-1,1)], dim=1)

        
        return gen_features
            
    def get_edge_features(self, edge_data, damages, node_data_pre, scenario, i, n_minus_k):
        #Edge Data
        #Feature Data
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
        #print(init_dmg[1806:1818])
        
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
                #print(3, j, len(adj_from), int(bus_from[j]), int(bus_to[j]), int(status[j]))
    
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
            impedance = torch.transpose(impedance, 0, 1).contiguous() #(5154, 2)
            impedance_complex = torch.view_as_complex(impedance) #(5154)
            admittance_complex = torch.reciprocal(impedance_complex) #(5154)
            
            edge_attr = - torch.view_as_real(admittance_complex)

            Y = torch.zeros((2000,2000), dtype=torch.cfloat)
            for idx, edge in enumerate(adj.t().tolist()):                
                source, target = edge
                Y[source, target] = admittance_complex[idx]
            #print('Y', Y)
            admittance_sum = torch.sum(Y, dim=0) #(2000), contains (y12 + y13, y12 + y23, ...)
            
            self_admittance = torch.cat((Gs.unsqueeze(1), Bs.unsqueeze(1)), dim=1) - torch.view_as_real(admittance_sum)
            #self_admittance = self_admittance + torch.view_as_real(admittance_sum) #DO POPRAWKI
            
            edge_attr = torch.cat([edge_attr, self_admittance], dim=0)
            edge_attr = torch.transpose(edge_attr, 0, 1)
            edge_attr = torch.tensor(np.sqrt(edge_attr[0,:]**2+edge_attr[1,:]**2))

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
                print(j)
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
        if name.startswith('./processed'):
            name=name[17:]
        else:
            name=name[21:]
        i=1
        while name[i].isnumeric():
            i+=1
        scenario=int(name[0:i])
        
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
        scenario=int(self.data_list[idx,0])
        step=int(self.data_list[idx,1])
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'
                                       f'_{step}.pt'))
        #print('HARDCODED MANUALLY REMOVING 4TH FEATURE IN X -> IN ORDER TO REMOVE VOLTAGE ANGLE WITHOUT REPROCESSING EVERYTHING')
        #data.x = torch.cat([data.x[:3],data.x[4:]])
        #print(data.x.shape)
        if self.embedding != None:
            #embedding = torch.cat([self.embedding]*int(len(data.x)/2000))
            #print(f'Embedding shape: {embedding.shape}')
            #print(f'self.embedding shape: {self.embedding.shape}')
            data.x = torch.cat([data.x.to('cpu'), self.embedding.to('cpu')], dim=1)
        return data
    
"""def collate_lstm(batch):    #Used for LSTM 
    for instance in batch:
        print('Pre Collate x:\n', instance.x.shape)
        print('Pre Collate edge_index:\n', instance.edge_index)
        print('Pre Collate edge_index shape:', instance.edge_index.shape)
        print('Pre Collate edge_attr:\n', instance.edge_attr)
        print('Pre Collate y:', instance.y)
    x = pad_sequence([a.x for a in batch]).permute(1,0,2)


    adj = pad_sequence([a.edge_index.permute(1,0) for a in batch]).permute(1,2,0)
    edge_attr = pad_sequence([a.edge_attr for a in batch]).permute(1,0,2)
    print('Post Collate x shape:', x.shape)
 
    print('Post Collate x:\n', x)
    print('Post Collate edge_index shape:', adj.shape)
    print('Post Collate edge_index:\n', adj)
    print('Post Collate edge_attr:\n', edge_attr)
    print('Post Collate y:', torch.tensor([a.y for a in batch]))
    data = Data(x=x, edge_index=adj, edge_attr=edge_attr, y=torch.tensor([a.y for a in batch]), batch_size=len(batch)) 

    return batch[0]"""
    

def create_datasets(root ,cfg, pre_transform=None, num_samples=None, stormsplit=0, embedding=None, data_type = 'AC', edge_attr='multi'):
    """
    Helper function which loads the dataset, applies
    pre-transforms and splits it into a training and a
    testing set.
    Input:
        root (str) : the root folder for the dataset
        pre_transform : pre-transform functions to be
            applied to the dataset, can be None
        div (float) : the ratio of training samples with
            respect to the entire dataset
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
    data_list = dataset.data_list

    if num_samples is None:
        len_dataset=len(dataset)
    else:
        print("Error: create_datasets can not accept num_samples as input yet")
    print(f'Len Dataset: {len_dataset}')
    #Get last train sample if stormsplit
    if stormsplit != 0:
        for i in range(len(data_list)):
            if str(data_list[i,0]).startswith(str(stormsplit)):
                last_train_sample=i
                break

    #Get last train sample if no stormsplit
    else:   
        trainsize = cfg["train_size"]
        last_train_sample = int(len_dataset*trainsize)
        if trainsize <1:
            while data_list[last_train_sample-1,0]==data_list[last_train_sample,0]:
                last_train_sample+=1
            testset = Subset(dataset, range(last_train_sample, len_dataset))
        else: testset= Subset(dataset,range(len_dataset,len_dataset))
    
    trainset = Subset(dataset, range(0, last_train_sample))
    testset = Subset(dataset, range(last_train_sample, len_dataset))
    
    t2 = time.time()
    print(f'Creating datasets took {(t1-t2)/60} mins', flush=True)

    return trainset, testset, data_list 

def create_loaders(cfg, trainset, testset, pre_compute_mean=False, Node2Vec=False, data_type='AC', num_workers=0, pin_memory=False): 
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

    if Node2Vec:
        trainloader = DataLoader(trainset, batch_size=1, shuffle=cfg["train_set::shuffle"]        
        )
        """elif data_type == 'LSTM':
            trainloader = DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], collate_fn=collate_lstm)
            testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"], collate_fn=collate_lstm)"""
    else:
        trainloader = DataLoader(trainset, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
    
    testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"], num_workers=num_workers, pin_memory=pin_memory)

    if pre_compute_mean:
        mean_labels = 0.
        for batch in testloader:   
            mean_labels += batch.y.sum().item()
        mean_labels /= len(testloader) 
        testloader.mean_labels = mean_labels

    print(f'Creating dataloaders took {(t1-time.time())/60} mins')
    return trainloader, testloader




def calc_mask_probs(dataloader):  
    """
    Calculates the masking probabilities based on the variance of the node
    
    Parameters
    ----------
    dataloader : the dataloader for the dataset for which the masking probabilities should be calculated

    Returns
    -------
    node_label_probs : float array
        the masking probabilities

    """

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
    return node_label_probs



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
    