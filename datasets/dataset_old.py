"""
Author : Jan Philipp Bohl

File setting up training and test data as well as neural network models
"""
import os
import logging
from math import floor
import numpy as np

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


import torch
from torch.utils.data import Subset


# Custom datasets

class HurricaneDataset(Dataset):
    """
    Custom class for the hurricane dataset
    """

    
    def __init__(self, root,use_supernode, transform=None, pre_transform=None, pre_filter=None):
        self.use_supernode=use_supernode
        super().__init__(root, transform, pre_transform, pre_filter)

        

        
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

        return ys.max()
    #TO
    def get_min_max_features(self):
        x_max=torch.zeros(2)
        x_min=torch.zeros(2)
        edge_attr_max=torch.zeros(1)
        edge_attr_min=torch.zeros(1)
        for i, raw_path in enumerate(self.raw_paths):
            x = torch.from_numpy(np.load(raw_path)["x"])
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
                
            edge_attr=torch.from_numpy(np.load(raw_path)["edge_weights"])
            for i in range(len(edge_attr)):
                if edge_attr[i]>edge_attr_max: edge_attr_max=edge_attr[i]
                if edge_attr[i]<edge_attr_min: edge_attr_min=edge_attr[i]
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
        idx = 0
        x_min, x_max, edge_attr_min, edge_attr_max = self.get_min_max_features()
        ymax=self.get_max_label()
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_data = np.load(raw_path)
            x = torch.from_numpy(raw_data["x"]).float()
            
            #TO
            #scaling

            x[0]=x[0]/(x_max[0]-x_min[0])
            x[1]=x[1]/(x_max[1]-x_min[1])
            
            #supernode
            print(self.use_supernode)
            if self.use_supernode:

                x=torch.cat((x,torch.tensor([[1,1]])),0)
            #TO end
            
            y = torch.tensor([raw_data["y"].item()]).float()/ymax
            adj = torch.from_numpy(raw_data["adj"])
   
            edge_attr = torch.from_numpy(raw_data["edge_weights"])
            
            #TO
            #scaling
            edge_attr=edge_attr/(edge_attr_max-edge_attr_min)
            #TO END
            
            adj, edge_attr = to_undirected(adj, edge_attr)

            #TO
            if self.use_supernode:
                Nnodes=len(x)
                supernode_edges=torch.zeros(2,Nnodes-1)
                for i in range(len(x)-1):
                    supernode_edges[0,i]=i
                    supernode_edges[1,i]=Nnodes-1
                adj = torch.cat((adj,supernode_edges),1)
                supernode_edge_attr=torch.zeros(supernode_edges.shape[1])+1
                edge_attr=torch.cat((edge_attr,supernode_edge_attr),0)
            #TO end

            data = Data(x=x, y=y, edge_index=adj, edge_attr=edge_attr)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
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
    dataset = HurricaneDataset(root=root,use_supernode=cfg["supernode"], pre_transform=pre_transform)

    if num_samples is None:
        len_dataset = len(dataset)
    else:
        len_dataset = num_samples

    last_train_sample = floor(cfg["train_size"] * len_dataset)

    trainset = Subset(dataset, range(0, last_train_sample))
    testset = Subset(dataset, range(last_train_sample, len_dataset))

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
        for batch in testloader:
            mean_labels += batch.y.sum().item()
        mean_labels /= len(testset)
        testloader.mean_labels = mean_labels

    logging.debug(f"Trainset first batch labels: {next(iter(trainloader)).y.tolist()}")
    logging.debug(f"Testset first batch labels: {next(iter(testloader)).y.tolist()}")

    return trainloader, testloader
