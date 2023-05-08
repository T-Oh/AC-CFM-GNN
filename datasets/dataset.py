"""
Author : Jan Philipp Bohl

File setting up training and test data as well as neural network models
"""
import os
import logging
from math import floor
import numpy as np
import scipy.io

from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


import torch
from torch.utils.data import Subset






#######################

# Custom datasets

class HurricaneDataset(Dataset):
    """
    Custom class for the hurricane dataset
    """

    
    def __init__(self, root,use_supernode, transform=None, pre_transform=None, pre_filter=None,N_Scenarios=100, stormsplit=0):
        self.use_supernode=use_supernode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.stormsplit = stormsplit
        self.data_list=self.get_data_list(N_Scenarios)
        print(self.data_list)
        
        
    
    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = os.listdir(self.root + "/processed")
        return [f for f in files if "data" in f]
    
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


                    
        #########################################################################################################################

        else:   
            test_idx = len(data_list)-1
            for file in self.processed_file_names:           
                if file.startswith(f'data'):
                    scenario, step = self.get_scenario_step_of_file(file)
                    if str(scenario).startswith(str(self.stormsplit)):
                        data_list[test_idx,:] = [scenario, step]
                        test_idx -= 1
                    else:
                        data_list[idx,:] = [scenario, step]
                        idx += 1
            
        return data_list
    
    
    def get_max_label(self):
        "Computes the max of all labels in the dataset"
        ys = np.zeros(len(self.raw_file_names))
        for i, raw_path in enumerate(self.raw_paths):
            ys[i] = torch.from_numpy(np.load(raw_path)["y"])

        return torch.tensor(ys.max())
    
    #TO
    def get_scenario_of_file(self,name):
        print(name)
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
                
    def get_scenario_info(self):
        #fills the array scenario with the number of steps each scenario contains
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
            length+=1
    
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
        x_max=torch.zeros(2)
        x_min=torch.zeros(2)
        x_means = torch.zeros(2)
        edge_attr_max=torch.zeros(5)
        edge_attr_min=torch.zeros(5)
        edge_attr_means = torch.zeros(5)
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
        
        
        for j, file in enumerate( self.processed_file_names):
            data = torch.load(self.processed_dir +'/' + file)
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
            
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                if edge_attr[i,0]>edge_attr_max[0]: edge_attr_max[0]=edge_attr[i,0]
                if edge_attr[i,0]<edge_attr_min[0]: edge_attr_min[0]=edge_attr[i,0]
                if edge_attr[i,1]>edge_attr_max[1]: edge_attr_max[1]=edge_attr[i,1]
                if edge_attr[i,1]<edge_attr_min[1]: edge_attr_min[1]=edge_attr[i,1]
                if edge_attr[i,2]>edge_attr_max[2]: edge_attr_max[2]=edge_attr[i,2]
                if edge_attr[i,2]<edge_attr_min[2]: edge_attr_min[2]=edge_attr[i,2]
                if edge_attr[i,4]>edge_attr_max[3]: edge_attr_max[3]=edge_attr[i,4]
                if edge_attr[i,4]<edge_attr_min[3]: edge_attr_min[3]=edge_attr[i,4]
                if edge_attr[i,5]>edge_attr_max[4]: edge_attr_max[4]=edge_attr[i,5]
                if edge_attr[i,5]<edge_attr_min[4]: edge_attr_min[4]=edge_attr[i,5]
                edge_attr_means[0] += edge_attr[i,0]
                edge_attr_means[1] += edge_attr[i,1]
                edge_attr_means[2] += edge_attr[i,2]
                edge_attr_means[3] += edge_attr[i,4]
                edge_attr_means[4] += edge_attr[i,5]
                edge_count += 1
                

            node_labels = data['node_labels']
            for i in range(len(node_labels)):
                if node_labels[i]>node_labels_max: node_labels_max=node_labels[i]
                if node_labels[i]<node_labels_min: node_labels_min=node_labels[i]
                node_labels_mean += node_labels[i]
            

        return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count
     
    def get_feature_stds(self, x_means, edge_means):
        x_stds = torch.zeros(2)
        edge_stds = torch.zeros(5)
        node_count = 0
        edge_count = 0
        for j, file in enumerate( self.processed_file_names):
            data = torch.load(self.processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                x_stds[0] += (x[i,0]-x_means[0])**2
                x_stds[1] += (x[i,1]-x_means[1])**2
                node_count += 1
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                edge_stds[0] += (edge_attr[i,0] - edge_means[0])**2
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
    
    def process(self):
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
        problems = [[0,0,0],[0,0,0]]
        #process data        
        for raw_path in self.raw_paths:
            #skip damage file
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
                if np.isnan(file['clusterresult'][0,i][21]):
                    #print('Skipping')
                    continue
                node_data_post = file['clusterresult'][0,i][2]   #node_data after step i for node_label_calculation
                P1 = node_data_pre[:,2] #P of all buses at initial condition - Node feature
                Q1 = node_data_pre[:,3] #Q of all buses at initial condition - Node feature
                S1 = np.sqrt(P1**2+Q1**2)
                Vm = node_data_pre[:,7] #Voltage magnitude of all buses at initial condition - Node feature
                #Va = node_data_pre[:,8] #Voltage angle of all buses at initial condition - Node feature
                
                P2 = node_data_post[:,2] #P of all buses after step - used for calculation of Node labels
                Q2 = node_data_post[:,3] #Q of all buses after step - used of calculation of Node labels
                S2 = np.sqrt(P2**2+Q2**2)
                
                node_labels = S1-S2
                
                
                #Edge Data
                #Feature Data
                rating = edge_data[:,5] #long term rating (MVA) - edge feature
                status = edge_data[:,10]  #1 if line is working 0 if line is not - edge feature
                resistance = edge_data[:,2]
                reactance = edge_data[:,3] 
                #power flows
                pf1 = edge_data[:,13]
                qf1 = edge_data[:,14]
                sf1 = np.sqrt(pf1**2+qf1**2)
                pf2 = edge_data[:,15]
                qf2 = edge_data[:,16]
                sf2 = np.sqrt(pf2**2+qf2**2)
                #Check for NaNs
                if any(np.isnan(pf1[status==1])) or any(np.isnan(qf1[status==1])) or any(np.isnan(pf2[status==1])) or any(np.isnan(qf2[status==1])):
                    print(raw_path)
                    for j in np.where(np.isnan(pf1))[0]:
                        if status[j]==1: 
                            print(j)
                            problems.append([scenario,i,j])
                    for j in np.where(np.isnan(qf1))[0]:
                        if status[j]==1: 
                            print(j)
                            problems.append([scenario,i,j])
                    for j in np.where(np.isnan(pf2))[0]:
                        if status[j]==1: 
                            print(j)
                            problems.append([scenario,i,j])
                    for j in np.where(np.isnan(qf2))[0]:
                        if status[j]==1: 
                            print(j)
                            problems.append([scenario,i,j])
                
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
                node_feature = torch.tensor([S1,Vm])
                node_labels = torch.tensor(node_labels)

                #save unscaled data
                data = Data(x=torch.transpose(node_feature,0,1).float(), edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1), node_labels=node_labels) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{scenario}_{i}.pt'))
            np.save('problems',np.array(problems))
        
        #SCALING
        #get limits for scaling
        """
        x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means = self.get_min_max_features()
        x_stds, edge_stds = self.get_feature_stds(x_means, edge_attr_means)
        
        for file in self.processed_file_names:
            data = torch.load(self.processed_dir + '/' + file)
            #Node features
            x = data['x']
            #node power
            x[:,0] = torch.log(x[:,0]+1)/torch.log(x_max[0]+1)
            #node voltage magnitude
            x[:,1] = ((x[:,1]-x_means[1])/x_stds[1])/((x_max[1]-x_means[1])/x_stds[1])
            
            #Edge Features
            edge_attr = data['edge_attr']
            #capacity
            edge_attr[:,0] = torch.log(data['edge_attr'][:,0]+1)/torch.log(edge_attr_max[0]+1)
            #Pf, QF and resistance
            edge_attr[:,1] = (data['edge_attr'][:,1]-edge_attr_means[1])/edge_stds[1]/((edge_attr_max[1]-edge_attr_means[1])/edge_stds[1])
            edge_attr[:,2] = (data['edge_attr'][:,2]-edge_attr_means[2])/edge_stds[2]/((edge_attr_max[2]-edge_attr_means[2])/edge_stds[2])
            edge_attr[:,4] = (data['edge_attr'][:,4]-edge_attr_means[3])/edge_stds[3]/((edge_attr_max[3]-edge_attr_means[3])/edge_stds[3])
            #reactance
            edge_attr[:,5] = torch.log(data['edge_attr'][:,5]+1)/torch.log(edge_attr_max[4]+1)
            
            #Node Labels
            node_labels = torch.log(data['node_labels']+1)/torch.log(node_labels_max+1)
            data = Data(x=x, edge_index=adj, edge_attr=edge_attr, node_labels=node_labels) 
            torch.save(data, os.path.join(self.processed_dir, file))"""
            
        
           

    def len(self):
        return len(self.processed_file_names)
    
    
    def get(self,idx):
        scenario=int(self.data_list[idx,0])
        step=int(self.data_list[idx,1])
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'
                                       f'_{step}.pt'))
        return data
    

def create_datasets(root ,cfg, pre_transform=None, num_samples=None):
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
    """
    dataset = HurricaneDataset(root=root,use_supernode=cfg["supernode"], pre_transform=pre_transform,N_Scenarios=cfg["n_scenarios"], stormsplit=cfg['stormsplit'])

    if num_samples is None:
        len_dataset=len(dataset)
    else:
        print("Error: create_datasets can not accept num_samples as input yet")
    print(f'Len Dataset: {len_dataset}')
    if cfg['stormsplit'] != 0:
        for i in range(len(dataset.data_list)):
            if str(dataset.data_list[i,0]).startswith('1'):
                last_train_sample=i-1
                break
    #last_train_sample = floor(trainsize * len_dataset)
    else:   
        trainsize = cfg["train_size"]
        last_train_sample = len_dataset*trainsize
        if trainsize <1:
            while dataset.data_list[last_train_sample-1,0]==dataset.data_list[last_train_sample,0]:
                last_train_sample+=1
            testset = Subset(dataset, range(last_train_sample, len_dataset))
        else: testset= Subset(dataset,range(len_dataset,len_dataset))
    
    trainset = Subset(dataset, range(0, last_train_sample))
    testset = Subset(dataset, range(last_train_sample, len_dataset))

    return trainset, testset

def create_loaders(cfg, trainset, testset, pre_compute_mean=False): 
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
    Return:
        trainloader : the training set loader
        testloader : the testing set loader
    """

    trainloader = DataLoader(trainset,
        batch_size=cfg["train_set::batchsize"],
        shuffle=cfg["train_set::shuffle"]
    )

    testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"])

    if pre_compute_mean:
        mean_labels = 0.
        for batch in testloader:   #change back to test
            mean_labels += batch.y.sum().item()
        mean_labels /= len(testloader) #change back to test
        testloader.mean_labels = mean_labels#change back to test

    #logging.debug(f"Trainset first batch labels: {next(iter(trainloader)).y.tolist()}")
    #logging.debug(f"Testset first batch labels: {next(iter(testloader)).y.tolist()}")

    return trainloader, testloader


def calc_mask_probs(dataloader):

    
    #New way
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
    mask_probs_rescaled = mask_probs.clone() + bias
    for i in range(len(mask_probs)):
        if mask_probs_rescaled[i] > 1.0: mask_probs_rescaled[i] = 1
    return mask_probs_rescaled
    