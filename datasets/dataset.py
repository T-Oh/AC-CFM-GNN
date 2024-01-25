"""
Author : Tobias Ohlinger

File setting up training and test data as well as neural network models
"""
import os
import numpy as np
import scipy.io
import time
import torch

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.utils import to_undirected






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

    
    def __init__(self, root,use_supernode, transform=None, pre_transform=None, pre_filter=None, N_Scenarios=100, stormsplit=0, embedding=None, device=None, data_type='AC'):
        self.use_supernode=use_supernode
        self.embedding = embedding
        self.device = device
        self.data_type = data_type
        super().__init__(root, transform, pre_transform, pre_filter)
        self.stormsplit = stormsplit
        self.data_list=self.get_data_list(N_Scenarios)  #list containing all instances in order
        print(self.data_list)
        
        
    
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
        if self.data_type == 'AC':
            self.process_ac()
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
        #load scenario file which stores the initial damages
        damages = self.get_initial_damages()
        #load initial network data
        init_data = scipy.io.loadmat('raw/' + 'pwsdata.mat')
        problems = [[0,0,0],[0,0,0]]    #used for error identification during processing
        #process data        
        for raw_path in self.raw_paths:
            #skip damage file and pws file 
            if 'Hurricane' in raw_path or 'pwsdata' in raw_path:
                continue
            scenario = self.get_scenario_of_file(raw_path)
            file=scipy.io.loadmat(raw_path)  #loads a full scenario   
            #loop through steps of scenario each step will be one processed data file
            for i in range(len(file['clusterresult'][0,:])):
                #Node data
                if i == 0:  #in first iteration load original pwsdata as initial data  
                    node_data_pre = init_data['ans'][0,0][2]    #ans is correct bcs its pwsdata
                    edge_data = init_data['ans'][0,0][4]
                else:
                    node_data_pre = file['clusterresult'][0,i-1][2]   #node_data of initial condition of step i
                    edge_data = file['clusterresult'][0,i-1][4] #edge data of initial condition of step i
                if np.isnan(file['clusterresult'][0,i][21]):    #This refers to matlab column ls_total -> if this is none the grid has failed completely in a previous iteration -> thus the data is invaluable and can be skipped
                    print('Skipping', file, i)
                    continue
                node_data_post = file['clusterresult'][0,i][2]   #node_data after step i for node_label_calculation
                
                node_feature, node_labels = self.get_node_features(node_data_pre, node_data_post)   #extract node features and labels from data
                
                adj, edge_attr, problems = self.get_edge_features(edge_data, damages, node_data_pre, scenario, i)
                problems.append(problems)
                
               
                
                #Graph Label
                graph_label = node_labels.sum()

                #save unscaled data
                data = Data(x=node_feature.float(), edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1), node_labels=node_labels, y=graph_label) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
            np.save('problems',np.array(problems))
    
        
    def get_node_features(self, node_data_pre, node_data_post):
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
        Va = torch.tensor(node_data_pre[:,8]) #Voltage angle of all buses at initial condition - Node feature
        
        P2 = torch.tensor(node_data_post[:,2]) #P of all buses after step - used for calculation of Node labels
        Q2 = torch.tensor(node_data_post[:,3]) #Q of all buses after step - used of calculation of Node labels
        S2 = torch.tensor(np.sqrt(P2**2+Q2**2))
        
        N_BUSES = len(node_data_pre[:,2])
        #one hot encoded bus types
        bus_type = torch.zeros([2000,4], dtype=torch.int32)
        for i in range(N_BUSES):
            bus_type[i, int(node_data_pre[i,1]-1)] = 1
        
        #one hot encoded node IDs
        node_ID = torch.eye(N_BUSES)
            
        
        node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), Va.reshape(-1,1), bus_type, node_ID], dim=1)
        node_labels = torch.tensor(S1-S2)
        
        
        
        return node_features, node_labels
            
    def get_edge_features(self, edge_data, damages, node_data_pre, scenario, i):
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
        
        problems = []
        
        if any(np.isnan(pf1[status==1])) or any(np.isnan(qf1[status==1])) or any(np.isnan(pf2[status==1])) or any(np.isnan(qf2[status==1])):
            problems.append(self.find_nans(status, pf1, scenario, i))
            problems.append(self.find_nans(status, pf2, scenario, i))
            problems.append(self.find_nans(status, qf1, scenario, i))
            problems.append(self.find_nans(status, qf2, scenario, i))

        
        #initial damages
        init_dmg = torch.zeros(len(status)) #edge feature that is 0 except if the line was an initial damage during that step
        #set initially damaged lines to 0
        for step in range(len(damages[scenario])):
            if damages[scenario][step,0] == i:
                status[damages[scenario][step,1]] = 0 
                init_dmg[damages[scenario][step,1]] = 1 
                
        #Adjacency Matrix
        bus_id = node_data_pre[:,0] #list of bus ids in order
        bus_from = edge_data[:,0]   
        bus_to = edge_data[:,1] 
        
        #Features
        adj_from = []   #adjacency matrix from/to -> no edges appearing twice
        adj_to = []
        rating_feature = [] #new list because orig data contains multiple lines along the same edge which are combined here
        status_feature = []
        resistance_feature = []
        reactance_feature = []
        init_dmg_feature = [] #always zero except if the line was an initial damage during this step -> then 1
        
        pf_feature = []
        qf_feature = []
        #Add edges and the respective features, edges are always added in both directions, so that the pf can be directional, the other features
        #are added to both directions
        for j in range(len(bus_from)):
            id_from = int(np.where(bus_id==bus_from[j])[0]) #bus_id where line starts
            id_to = int(np.where(bus_id==bus_to[j])[0])     #bus_id where line ends
            
            #if edge already exists recalculate (add) the features and dont add a new edge
            exists=False
            if (adj_from.count(id_from) > 0):   #check if bus from exists
                for k in range(len(adj_from)):  #check all appeareances of bus from in adj_from
                    if adj_from[k] == id_from and adj_to[k] == id_to: #if bus from and bus to are at the same entry update their edge features
                        exists = True                       #mark as edge exists
                        if status_feature[k] != 0:          #if status 0 keep 0 otherwise set to status of additional edge
                            status_feature[k] = status[j]
                        if status_feature[k] == 0:
                            rating_feature[k] = 0
                            resistance_feature[k] = 1
                            reactance_feature[k] = 1
                            pf_feature[k] = 0
                            qf_feature[k] = 0
                        else:
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
                        if status_feature[k] != 0:          #if status 0 keep 0 otherwise set to status of additional edge
                            status_feature[k] = status[j]
                        if status_feature[k] == 0:
                            rating_feature[k] = 0
                            resistance_feature[k] = 1
                            reactance_feature[k] = 1
                            pf_feature[k] = 0
                            qf_feature[k] = 0
                        else:
                            rating_feature[k] += rating[j]      #add the capacities (ratings)
                            resistance_feature[k], reactance_feature[k] = self.calc_total_resistance_reactance(resistance_feature[k],resistance[j],reactance_feature[k], reactance[j])
                            pf_feature[k] += qf2[j]
                            qf_feature[k] += qf2[j]

                        if init_dmg_feature[k] != 1:
                            init_dmg_feature[k] = init_dmg[j]
                        
            if exists: continue
            #if edge does not exist yet add it in both directions
            else:
                #First direction
                
                adj_from.append(id_from)
                adj_to.append(id_to)
    
                status_feature.append(status[j])
    
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
                status_feature.append(status[j])
                resistance_feature.append(resistance[j])
                reactance_feature.append(reactance[j])
                init_dmg_feature.append(init_dmg[j])
            

            

        #compile data of step to feature matrices                
        adj = torch.tensor([adj_from,adj_to])

        edge_attr = torch.tensor([rating_feature, pf_feature, qf_feature, status_feature, resistance_feature, reactance_feature, init_dmg_feature])
        
        return adj, edge_attr, problems
      
    def find_nans(status, feature, scenario, i):
        #Check for NaNs
        problems = []
        
        for j in np.where(np.isnan(feature))[0]:
            if status[j]==1: 
                print(j)
                problems.append([scenario,i,j])
        return problems

            
    def process_dc(self):
        """
        Loads the raw numpy data, converts it to torch
        tensors and normalizes the labels to lie in
        the interval [0,1].
        Then pre-filters and pre-transforms are applied
        and the data is saved in the processed directory.
        """
        #get limits for scaling

        x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means, graph_label_min, graph_label_max, graph_label_mean = self.get_min_max_features()
        x_stds, edge_stds = self.get_feature_stds(x_means, edge_attr_means)
        ymax=torch.log(self.get_max_label()+1)
        
        #process data
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_data = np.load(raw_path)
            
            #scale node features
            x = torch.from_numpy(raw_data["x"]).float()
            x[:,0]=(x[:,0]-x_means[0])/x_stds[0]/((x_max[0]-x_means[0])/x_stds[0])
            x[:,1]=(x[:,1]-x_means[1])/x_stds[1]/((x_max[1]-x_means[1])/x_stds[1])
            
            #add supernode
            if self.use_supernode:
                x=torch.cat((x,torch.tensor([[1,1]])),0)            
            
            #get and scale labels according to task (GraphReg or NodeReg)
            y = torch.log(torch.tensor([raw_data["y"].item()]).float()+1)/ymax
            if 'node_labels' in raw_data.keys():
                node_labels=torch.from_numpy(raw_data['node_labels'])
                node_labels= torch.log(node_labels+1)/torch.log(node_labels_max+1)
                node_labels.type(torch.FloatTensor)
            
            #get and scale edges and adjacency matrix
            adj = torch.from_numpy(raw_data["adj"])  
            print(adj)
            edge_attr = torch.from_numpy(raw_data["edge_weights"])

            if edge_attr.shape[0]<3:    #multidimensional edge features 
                edge_attr[0]=torch.log(edge_attr[0]+1)/torch.log(edge_attr_max+1)                
                adj, edge_attr = to_undirected(adj,[edge_attr[0].float(),edge_attr[1].float()])
                edge_attr=torch.stack(edge_attr)

                
            else:                
                #transform to undirected graph            
                adj, edge_attr = to_undirected(adj, edge_attr)
            

            #add supernode edges
            if self.use_supernode:
                Nnodes=len(x)
                supernode_edges=torch.zeros(2,Nnodes-1)
                for i in range(len(x)-1):
                    supernode_edges[0,i]=i
                    supernode_edges[1,i]=Nnodes-1
                adj = torch.cat((adj,supernode_edges),1)
                supernode_edge_attr=torch.zeros(supernode_edges.shape[1])+1
                edge_attr=torch.cat((edge_attr,supernode_edge_attr),0)
            

            print('Using Homogenous Data')
            if 'node_labels' in raw_data.keys():
                data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1),node_labels=node_labels)
            else:
                data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1))
            

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            print(data)
            #torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            torch.save(data, os.path.join(self.processed_dir, 'data_'+raw_path[15:-4]+'.pt'))        
           
    def get_max_label(self):
        "Computes the max of all labels in the dataset"
        ys = np.zeros(len(self.raw_file_names))
        for i, raw_path in enumerate(self.raw_paths):
            ys[i] = torch.from_numpy(np.load(raw_path)["y"])

        return torch.tensor(ys.max())
    
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
                
    '''def get_scenario_info(self):
        #fills the array scenario with the number of steps each scenario contains
        #deprecated ?
        scenario=0
        length=1
        for index,name in enumerate(self.raw_paths):
            if index==len(self.raw_paths)-1:
                self.scenarios[scenario]=length
                break;
            s1,_=self.get_scenario_step_of_file(name)
            s2,_=self.get_scenario_step_of_file(self.raw_paths[index+1])
            if s1!=s2: 
                self.scenarios[scenario]=length
                length=0
                scenario+=1
            length+=1'''
    
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
                    
    
    def get_min_max_features(self):
        """
        Returns the min and max of the features
        Probably deprecated as normalization is not part of dataset anymore
        """
        x_max=torch.zeros(2)
        x_min=torch.zeros(2)
        x_means = torch.zeros(2)
        edge_attr_max=0
        edge_attr_min=0
        edge_attr_means = 0
        node_count = 0
        edge_count = 0
        for i in range(5):
            edge_attr_max[i] =  np.NINF
            edge_attr_min[i] = np.Inf
            if i <2:
                x_max[i] = np.NINF
                x_min[i] = np.Inf
        node_labels_max=0
        node_labels_min=1e6
        node_labels_mean = 0
        graph_label_max = np.NINF
        graph_label_min = np.Inf
        graph_label_mean = 0
        graph_count = 0
        
        
        for file in self.raw_paths:
            # Read data from `raw_path`.
            
       
            graph_count += 1
            data = np.load(file)
            #data = torch.load(self.processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
                x_means[0] += x[i,0]
                x_means[1] += x[i,1]
                node_count += 1
                #if x[i,2]>x_max[2]: x_max[2]=x[i,2]    can be used for a third node feature
                #if x[i,2]<x_min[2]: x_min[2]=x[i,2]
            
            edge_attr = data['edge_weights']
            for i in range(len(edge_attr)):
                if self.data_type == 'DC':
                    if edge_attr[i]>edge_attr_max: edge_attr_max=edge_attr[i]
                    if edge_attr[i]<edge_attr_min: edge_attr_min=edge_attr[i]
                    edge_attr_means += edge_attr[i]
                if self.data_type == 'AC':
                    if edge_attr[i,0]>edge_attr_max[0]: edge_attr_max[0]=edge_attr[i,0]
                    if edge_attr[i,0]<edge_attr_min[0]: edge_attr_min[0]=edge_attr[i,0]
                    edge_attr_means[0] += edge_attr[i,0]
                    if edge_attr[i,1]>edge_attr_max[1]: edge_attr_max[1]=edge_attr[i,1]
                    if edge_attr[i,1]<edge_attr_min[1]: edge_attr_min[1]=edge_attr[i,1]
                    if edge_attr[i,2]>edge_attr_max[2]: edge_attr_max[2]=edge_attr[i,2]
                    if edge_attr[i,2]<edge_attr_min[2]: edge_attr_min[2]=edge_attr[i,2]
                    if edge_attr[i,4]>edge_attr_max[3]: edge_attr_max[3]=edge_attr[i,4]
                    if edge_attr[i,4]<edge_attr_min[3]: edge_attr_min[3]=edge_attr[i,4]
                    if edge_attr[i,5]>edge_attr_max[4]: edge_attr_max[4]=edge_attr[i,5]
                    if edge_attr[i,5]<edge_attr_min[4]: edge_attr_min[4]=edge_attr[i,5]

                    edge_attr_means[1] += edge_attr[i,1]
                    edge_attr_means[2] += edge_attr[i,2]
                    edge_attr_means[3] += edge_attr[i,4]
                    edge_attr_means[4] += edge_attr[i,5]
                edge_count += 1
            graph_label = data['y']
            if graph_label>graph_label_max: graph_label_max=graph_label
            if graph_label<graph_label_min: graph_label_min=graph_label
            graph_label_mean += graph_label
            
            if self.data_type == 'AC':
                node_labels = data['node_labels']
                for i in range(len(node_labels)):
                    if node_labels[i]>node_labels_max: node_labels_max=node_labels[i]
                    if node_labels[i]<node_labels_min: node_labels_min=node_labels[i]
                    node_labels_mean += node_labels[i]
            else: node_count = 1    #avoid devision by 0
            

        return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count, graph_label_min, graph_label_max, graph_label_mean/graph_count
     
    def get_feature_stds(self, x_means, edge_means):
        """
        Input 
        x_means     float array
            means of the node features
        edge_means  float array
            means of the edge features
            
        Returns
        np.sqrt(x_stds/node_count)
            the standarad deviations of the node features
        
        np.sqrt(edge_stds/edge_count)
            the standard deviations of the edge features
        """
        
        x_stds = torch.zeros(2)
        edge_stds = torch.zeros(5)
        node_count = 0
        edge_count = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = np.load(raw_path)
            #data = torch.load(self.processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                x_stds[0] += (x[i,0]-x_means[0])**2
                x_stds[1] += (x[i,1]-x_means[1])**2
                node_count += 1
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                edge_stds[0] += (edge_attr[i,0] - edge_means[0])**2
                if self.data_type == 'AC':
                    edge_stds[1] += (edge_attr[i,1] - edge_means[1])**2
                    edge_stds[2] += (edge_attr[i,2] - edge_means[2])**2
                    edge_stds[3] += (edge_attr[i,4] - edge_means[3])**2
                    edge_stds[4] += (edge_attr[i,5] - edge_means[4])**2
                edge_count += 1
        return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count)
                
        
            

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
    
    
    def get(self,idx):
        scenario=int(self.data_list[idx,0])
        step=int(self.data_list[idx,1])
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'
                                       f'_{step}.pt'))
        print(data.x.shape)
        if self.embedding != None:
            print('YES')
            #embedding = torch.cat([self.embedding]*int(len(data.x)/2000))
            #print(f'Embedding shape: {embedding.shape}')
            print(f'self.embedding shape: {self.embedding.shape}')
            data.x = torch.cat([data.x.to('cpu'), self.embedding.to('cpu')], dim=1)
        return data
    

def create_datasets(root ,cfg, pre_transform=None, num_samples=None, stormsplit=0, embedding=None, data_type = 'AC'):
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
    dataset = HurricaneDataset(root=root,use_supernode=cfg["supernode"], pre_transform=pre_transform,N_Scenarios=cfg["n_scenarios"], stormsplit=stormsplit, embedding=embedding, data_type=data_type)
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
    print(f'Creating datasets took {(t1-t2)/60} mins')

    return trainset, testset, data_list 

def create_loaders(cfg, trainset, testset, pre_compute_mean=False, Node2Vec=False): 
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
        trainloader = DataLoader(trainset,
            batch_size=1,
            shuffle=cfg["train_set::shuffle"]
        )
    else:
        trainloader = DataLoader(trainset,
            batch_size=cfg["train_set::batchsize"],
            shuffle=cfg["train_set::shuffle"]
        )

    testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"])

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
    