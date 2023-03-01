"""
Author : Jan Philipp Bohl

File setting up training and test data as well as neural network models
"""
import os
import logging
from math import floor
import numpy as np

from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


import torch
from torch.utils.data import Subset


# Custom datasets

class HurricaneDataset(Dataset):
    """
    Custom class for the hurricane dataset
    """

    
    def __init__(self, root,use_supernode, transform=None, pre_transform=None, pre_filter=None,N_Scenarios=0):
        self.use_supernode=use_supernode
        #self.scenarios=np.zeros(N_Scenarios)
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.get_scenario_info()
        self.data_list=self.get_data_list(N_Scenarios)
        print(self.data_list)
        
    
    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = os.listdir(self.root + "/processed")
        return [f for f in files if "data" in f]

    def get_max_label(self):
        "Computes the max of all labels in the dataset"
        ys = np.zeros(len(self.raw_file_names))
        for i, raw_path in enumerate(self.raw_paths):
            ys[i] = torch.from_numpy(np.load(raw_path)["y"])

        return torch.tensor(ys.max())
    
    #TO
    def get_scenario_step_of_file(self,name):       
        name=name[15:]
        i=1
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
    
    def get_data_list(self,N_scenarios):
        #N_scenario must be last Scenario that appears in raw (if scenario 1,2 and 100 are used N_scenarios must be 100)
        data_list=np.zeros((len(self.raw_paths),2))
        idx=0                           
        for i in range(N_scenarios):
            first=True
            for file in self.raw_paths:
                if file.startswith(f'.\\raw\\scenario_{i+1}_'):
                    
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
        return data_list
                    
    
    def get_min_max_features(self):
        x_max=torch.zeros(2)
        x_min=torch.zeros(2)
        edge_attr_max=torch.zeros(1)
        edge_attr_min=torch.zeros(1)
        node_labels_max=0
        node_labels_min=1e6
        for j, raw_path in enumerate(self.raw_paths):
            x = torch.from_numpy(np.load(raw_path)["x"])
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
            
            edge_attr=torch.from_numpy(np.load(raw_path)["edge_weights"])[0]
            for i in range(len(edge_attr)):
                if edge_attr[i]>edge_attr_max: edge_attr_max=edge_attr[i]
                if edge_attr[i]<edge_attr_min: edge_attr_min=edge_attr[i]
                
            if 'node_labels' in np.load(raw_path).keys():
                print('TEST')
                node_labels=torch.from_numpy(np.load(raw_path)['node_labels'])
                for i in range(len(node_labels)):
                    if node_labels[i]>node_labels_max: node_labels_max=node_labels[i]
                    if node_labels[i]<node_labels_min: node_labels_min=node_labels[i]
                return x_min,x_max,edge_attr_min,edge_attr_max,node_labels_min, node_labels_max
            else:
                return x_min,x_max,edge_attr_min,edge_attr_max
        
            

    def download(self):
        pass


    def process(self):
        """
        Loads the raw numpy data, converts it to torch
        tensors and normalizes the labels to lie in
        the interval [0,1].
        Then pre-filters and pre-transforms are applied
        and the data is saved in the processed directory.
        """
        #get limits for scaling
        x_min, x_max, edge_attr_min, edge_attr_max, node_labels_min, node_labels_max = self.get_min_max_features()
        #x_min, x_max, edge_attr_min, edge_attr_max = self.get_min_max_features()
        ymax=torch.log(self.get_max_label()+1)
        
        #process data
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_data = np.load(raw_path)
            
            #scale node features
            x = torch.from_numpy(raw_data["x"]).float()
            x[:,0]=x[:,0]/(x_max[0]-x_min[0])
            x[:,1]=x[:,1]/(x_max[1]-x_min[1])
            
            #add supernode
            if self.use_supernode:
                x=torch.cat((x,torch.tensor([[1,1]])),0)            
            
            #get and scale labels according to task (GraphReg or NodeReg)
            y = torch.log(torch.tensor([raw_data["y"].item()]).float()+1)/ymax
            if 'node_labels' in raw_data.keys():
                node_labels=torch.from_numpy(raw_data['node_labels'])
                node_labels= (node_labels-node_labels_min)/(node_labels_max-node_labels_min)
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
            
            #compile Data
            if len(edge_attr)<3:    #onedimensional edge features
                print('Using Homogenous Data')
                if 'node_labels' in raw_data.keys():
                    data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1),node_labels=node_labels)
                else:
                    data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1))
            else:   #multidimensional edge features
                data=HeteroData()
                data['bus'].x=x
                
                data['bus','line','bus'].edge_index=adj
                
                data['bus','line','bus'].edge_attr=torch.transpose(edge_attr,0,1)
                
                data.y=y

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            print(data)
            #torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            torch.save(data, os.path.join(self.processed_dir, 'data_'+raw_path[15:-4]+'.pt'))
            

    def len(self):
        return len(self.raw_file_names)
    
    
    def get(self,idx):
        scenario=int(self.data_list[idx,0])
        step=int(self.data_list[idx,1])
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'
                                       f'_{step}.pt'))
        return data
    
    """def get(self, idx):
        if idx==0:
            data = torch.load(os.path.join(self.processed_dir, f'data_1_1.pt'))
            return data 
                   
        scenario=0
        i=0
        acc_scenarios=0
        while idx>acc_scenarios:
            scenario+=1
            acc_scenarios+=self.scenarios[i]
            i+=1
        acc_scenarios=acc_scenarios-self.scenarios[i-1]
        step=int(idx-acc_scenarios)
        
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'
                                       f'_{step}.pt'))
        return data"""

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
    dataset = HurricaneDataset(root=root,use_supernode=cfg["supernode"], pre_transform=pre_transform,N_Scenarios=cfg["n_scenarios"])

    if num_samples is None:
        len_dataset=len(dataset)
    else:
        print("Error: create_datasets can not accept num_samples as input yet")
        
    trainsize = cfg["train_size"]
    last_train_sample = floor(trainsize * len_dataset)
    if trainsize <1:
        while dataset.data_list[last_train_sample-1,0]==dataset.data_list[last_train_sample,0]:
            last_train_sample+=1
        testset = Subset(dataset, range(last_train_sample, len_dataset))
    else: testset= Subset(dataset,range(len_dataset,len_dataset))
    trainset = Subset(dataset, range(0, last_train_sample))
    

    return trainset, testset

def create_loaders(cfg, trainset, testset, pre_compute_mean=True): 
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
        for batch in trainloader:   #change back to test
            mean_labels += batch.y.sum().item()
        mean_labels /= len(trainloader) #change back to test
        trainloader.mean_labels = mean_labels#change back to test

    logging.debug(f"Trainset first batch labels: {next(iter(trainloader)).y.tolist()}")
    #logging.debug(f"Testset first batch labels: {next(iter(testloader)).y.tolist()}")

    return trainloader, testloader
