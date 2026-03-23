import os
import numpy as np
import scipy.io
import time
import torch
import json

from os.path import isfile


from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, random_split
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
    data_type       the type of data to be processed (AC, LSTM, Zhu, Zhu_mat73, ANGF_Vcf, Zhu_nobustype, Zhu_n_minus_k, LDTSF, LDTSF_DC, n-k)
    """

    
    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            N_Scenarios=100,
            stormsplit=0,
            embedding=None,
            device=None,
            data_type='AC',
            edge_attr='multi',
            ls_threshold = .09,
            N_below_threshold=1,
            normalize_injection=True,
            multiply_base_voltage=False,
            zhu_check_buses=False,
            check_s_y=False,
            normalized=True
    ):
        #self.use_supernode=use_supernode
        self.root = root
        self.PROCESSING_LSTM_DATA = False 
        self.embedding = embedding
        self.device = device
        self.data_type = data_type
        self.edge_attr = edge_attr
        self.ls_threshold = ls_threshold
        self.N_below_threshold = N_below_threshold
        self.normalized = normalized

        self.normalize_injection = normalize_injection
        self.multiply_base_voltage = multiply_base_voltage
        self.zhu_check_buses = zhu_check_buses
        self.check_s_y = check_s_y
        super().__init__(root, transform, pre_transform, pre_filter)
        self.stormsplit = stormsplit
        if self.data_type != 'LSTM':
            self.data_list=self.get_data_list(N_Scenarios)  #list containing all instances in order
        print('End of init')
        print(self.PROCESSING_LSTM_DATA)

        
        
    
    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = []
        if self.normalized:
            for root, _, filenames in os.walk(os.path.join(self.root, "normalized")):
                for filename in filenames:
                    if filename.startswith("data"):
                        files.append(os.path.relpath(os.path.join(root, filename), self.root + "/normalized"))
        else:
            for root, _, filenames in os.walk(os.path.join(self.root, "processed")):
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
                if file.startswith('data') and file.endswith('.pt'):
                    scenario, step = self.get_scenario_step_of_file(file)
                    data_list[idx,:] = [scenario, step]                   
                    idx += 1       
        #Stormsplit
        else:   
            test_idx = len(data_list)-1
            for file in self.processed_file_names:           
                if file.startswith('data') and file.endswith('.pt'):
                    scenario, step = self.get_scenario_step_of_file(file)
                    if str(scenario).startswith(str(self.stormsplit)):
                        data_list[test_idx,:] = [scenario, step]
                        test_idx -= 1
                    else:
                        data_list[idx,:] = [scenario, step]
                        idx += 1
            
        return data_list
    
 
    def find_nans(status, feature, scenario, i):
        #Check for NaNs
        problems = []
        
        for j in np.where(np.isnan(feature))[0]:
            if status[j]==1: 
                problems.append([scenario,i,j])
        return problems


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
                
    
   

    def download(self):
        pass   
    
    def len(self):
        return len(self.processed_file_names)
    
    
    def __getitem__(self,idx):
        scenario=int(self.data_list[idx,0])
        step=int(self.data_list[idx,1])
        data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'f'_{step}.pt'))
        #if self.data_type == 'LDTSF':
        #    scenario=int(self.data_list[idx,0])
        #    step=int(self.data_list[idx,1])
        #    data = torch.load(os.path.join(self.processed_dir, f'data_{scenario}'f'_{step}.pt'))
        #else:
        #    data = torch.load(os.path.join(self.processed_dir, self.data_list[idx]))

        #QUESTION
        if 'zhu' not in self.data_type.lower():
                data.x = data.x[:, :4]

        return data
    
  
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
        padded_targets[i, -lengths[i]:] = torch.tensor(targets[i])  # Place the sequence at the end, leaving padding at the start
        padded_targets_class[i, -lengths[i]:] = torch.tensor(targets_class[i])
    lengths[:] = max_len


    return padded_sequences, padded_targets, lengths, padded_targets_class

    

def create_datasets(cfg, embedding=None, normalized=True):
    """
    Helper function which loads the dataset and splits it into a training and a
    testing set.
    Input:
        root (str) : the root folder for the dataset
        normalized (bool) : whether to use normalized data
    Return:
        trainset : the training set
        testset : the testset
        data_list : the data_list
    """
    print('Creating Datasets...')
    t1 = time.time()
    stormsplit = cfg['stormsplit']
    dataset = HurricaneDataset(
        root=cfg['dataset::path'],
        N_Scenarios=cfg["n_scenarios"],
        stormsplit=stormsplit,
        embedding=embedding,
        data_type=cfg['data'],
        edge_attr=cfg['edge_attr'],
        ls_threshold=cfg['ls_threshold'],
        N_below_threshold=cfg['N_below_threshold'],
        normalize_injection=cfg['normalize_injection'],
        multiply_base_voltage=cfg['multiply_base_voltage'],
        zhu_check_buses=cfg['zhu_check_buses'],
        check_s_y=cfg['check_s_y'],
        normalized=normalized
    )
    #data_list = dataset.data_list
    print('create_dataset()')
    print('PROCESSING_LSTM_DATA:', dataset.PROCESSING_LSTM_DATA)
    if dataset.PROCESSING_LSTM_DATA:
        t2 = time.time()
        print(f'Processing took {(t2-t1)/60} mins', flush=True)
        return None, None, dataset.PROCESSING_LSTM_DATA

    len_dataset = len(dataset)
    print(f'Len Dataset: {len_dataset}')
    #Get last train sample if stormsplit
    if stormsplit != 0:
        print(len(dataset.data_list))
        for i in range(len(dataset.data_list)):
            print(dataset.data_list[i][0])
            if str(dataset.data_list[i][0]).startswith(str(stormsplit)):
                last_train_sample=i-1
                break
        trainset = Subset(dataset, range(0, last_train_sample))
        testset = Subset(dataset, range(last_train_sample, len_dataset))

    #Get last train sample if no stormsplit
    else:
        trainsize = cfg["train_size"]
        last_train_sample = int(len_dataset*trainsize)

        trainset, testset = random_split(dataset, [last_train_sample, len_dataset-last_train_sample])
    
    t2 = time.time()
    print(f'Creating datasets took {(t2-t1)/60} mins', flush=True)

    return trainset, testset, False

def create_datasets_zhu(
        root,
        cfg,
        pre_transform=None,
        num_samples=None,
        stormsplit=0,
        embedding=None,
        data_type = 'AC',
        edge_attr='multi',
        normalize_injection=True,
        multiply_base_voltage=False,
        zhu_check_buses=False,
        check_s_y=False
):
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
    dataset = HurricaneDataset(
        root=root,
        use_supernode=cfg["supernode"],
        pre_transform=pre_transform,
        N_Scenarios=cfg["n_scenarios"],
        stormsplit=stormsplit,
        embedding=embedding,
        data_type=data_type,
        edge_attr=edge_attr,
        ls_threshold=cfg['ls_threshold'],
        N_below_threshold=cfg['N_below_threshold'],
        normalize_injection=normalize_injection,
        multiply_base_voltage=multiply_base_voltage,
        zhu_check_buses=zhu_check_buses,
        check_s_y=check_s_y
    )
    data_list = dataset.data_list

    if num_samples is None:
        len_dataset = len(dataset)
    else:
        print("Error: create_datasets can not accept num_samples as input yet")
    print(f'Len Dataset: {len_dataset}')
    # Get last train sample if stormsplit
    if stormsplit != 0:
        for i in range(len(data_list)):
            if str(data_list[i, 0]).startswith(str(stormsplit)):
                last_train_sample = i
                break

    # Get last train sample if no stormsplit
    else:
        if cfg['pl_stage'] == 'train':  # simply training with MSE loss, without pretraining with physics loss
            trainsize = cfg["train_size"]
            last_train_sample = int(len_dataset * trainsize)
            if trainsize < 1:
                while data_list[last_train_sample - 1, 0] == data_list[last_train_sample, 0]:
                    last_train_sample += 1
                testset = Subset(dataset, range(last_train_sample, len_dataset))
            else:
                testset = Subset(dataset, range(len_dataset, len_dataset))

        else:
            trainsize = cfg["train_size"]
            finetune_size = cfg["train_size"]

            last_train_sample = int(len_dataset * trainsize)
            last_finetune_sample = last_train_sample + int((len_dataset - last_train_sample) * finetune_size)
            '''
            if trainsize <1:
                while data_list[last_train_sample-1,0]==data_list[last_train_sample,0]:
                    #print('dataset.py data list', data_list[last_train_sample-1,0])
                    last_train_sample+=1
                testset = Subset(dataset, range(last_train_sample, len_dataset))
            else: testset= Subset(dataset,range(len_dataset,len_dataset))
            '''
            pretrain_set = Subset(dataset, range(0, last_train_sample))
            finetune_set = Subset(dataset, range(last_train_sample, last_finetune_sample))
            testset = Subset(dataset, range(last_train_sample, len_dataset))

    if last_train_sample == len_dataset:
        testset = pretrain_set
        print('testset = pretrain_set !')
    # print('trainset', pretrain_set)
    # print('testset', testset)
    # print('last train sample, len_dataset', last_train_sample, len_dataset)
    t2 = time.time()
    print(f'Creating datasets took {(t1 - t2) / 60} mins', flush=True)

    return pretrain_set, finetune_set, testset, data_list

def create_loaders_zhu(cfg, pretrain_set, finetune_set, testset, pre_compute_mean=False, Node2Vec=False, data_type='AC',
                   num_workers=0, pin_memory=False):
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

    pretrain_loader = DataLoader(pretrain_set, batch_size=cfg["train_set::batchsize"],
                                 shuffle=cfg["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
    finetune_loader = DataLoader(finetune_set, batch_size=cfg["train_set::batchsize"],
                                 shuffle=cfg["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=cfg["test_set::batchsize"], num_workers=num_workers,
                            pin_memory=pin_memory)

    # for i, batch in enumerate(pretrain_loader):
    # print(f"Batch {i}: {type(batch)}")
    #    for key, value in batch.items():
    # print(f"{key}: {value.size()}")

    if pre_compute_mean:
        mean_labels = 0.
        for batch in testloader:
            mean_labels += batch.y.sum().item()
        mean_labels /= len(testloader)
        testloader.mean_labels = mean_labels

    print(f'Creating dataloaders took {(t1 - time.time()) / 60} mins')
    return pretrain_loader, finetune_loader, testloader

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
        max_length = lstm_get_max_seq_length(trainset, testset)
        print(max_length)
        if 'typeII' in task:         

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





    

