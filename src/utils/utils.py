import torch
import json
import os

import scipy.io
import torch.nn

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from ray import tune
import warnings
from torch_geometric.data import Data
from torch_scatter import scatter_add
from datasets.dataset import create_datasets, create_loaders
from datasets.dataset_graphlstm import create_lstm_datasets, create_lstm_dataloader


def check_config_conflicts(cfg):
    assert not (cfg['crossvalidation'] and cfg['study::run']),  'can only run a study or the crossvalidation not both'
    assert not (cfg['data'] == 'DC' and cfg['stormsplit']>0),   'Stormsplit can only be used with AC data'
    assert not (cfg['edge_attr'] == 'multi' and cfg['model'] == 'TAG'), 'TAG can only be used with Y as edge_attr not with multi'
    assert not (cfg['data'] == 'LDTSF' and cfg['task'] == 'NodeReg'),   'LDTSF Only works with GraphReg and GraphClass'
    assert not (cfg['data'] == 'LDTSF' and cfg['model'] != 'lstm'),     'LDTSF Only works with lstm as model'
    assert not (cfg['data'] == 'AC' and cfg['task'] == 'GraphClass'),   'None of the models working with AC data has GraphClass implemented' 
    if cfg['data'] == 'LSTM' and not cfg['model'] == 'GATLSTM': 
        warnings.warn("Using LSTM data with a model that is not GATLSTM, this should only be done for processing of LSTM data", UserWarning)
    

def setup_searchspace(cfg):
    """
    Sets up the searchspace for a study based on the configuration file

    Parameters
    ----------
    cfg : preloaded json configuration file


    Returns
    -------
    search_space :dictrionary
        dictionary of the search space used by ray

    """

    search_space = {}
    #General Architecture
    if cfg['study::lr::lower'] != cfg['study::lr::upper']:
        search_space['LR'] = tune.uniform(cfg['study::lr::lower'], cfg['study::lr::upper'])
    if cfg['study::weight_decay_upper'] != cfg['study::weight_decay_lower']:
        search_space['weight_decay'] = tune.uniform(cfg['study::weight_decay_lower'], cfg['study::weight_decay_upper'])

    if cfg["study::layers_lower"] != cfg["study::layers_upper"]:
        search_space['num_layers'] = tune.quniform(cfg["study::layers_lower"], cfg["study::layers_upper"]+1, 1)
    if cfg["study::hidden_features_lower"] != cfg["study::hidden_features_upper"]:
        search_space['hidden_size'] = tune.loguniform(cfg["study::hidden_features_lower"], cfg["study::hidden_features_upper"]+1)
    if cfg["study::dropout_lower"] != cfg["study::dropout_upper"]:
        search_space['dropout'] = tune.quniform(cfg["study::dropout_lower"], cfg["study::dropout_upper"], 0.01)
    if cfg['study::skipcon']:
        search_space['use_skipcon'] = tune.uniform(0, 2)
    if cfg['study::batchnorm']:
        search_space['use_batchnorm'] = tune.uniform(0, 2)

    #Regression Head
    if cfg['study::reghead_size_lower'] != cfg['study::reghead_size_upper']:
        search_space['reghead_size'] = tune.loguniform(cfg['study::reghead_size_lower'], cfg['study::reghead_size_upper']+1)
    if cfg["study::reghead_layers_lower"] != cfg['study::reghead_layers_upper']:
        search_space['reghead_layers'] = tune.uniform(cfg["study::reghead_layers_lower"], cfg['study::reghead_layers_upper']+1)
    if cfg["study::reghead_type"]:
        search_space['reghead_type'] = tune.uniform(0, 3)

    #LSTM Layers
    if cfg['study::num_conv_targets_lower'] != cfg['study::num_conv_targets_upper']:
        search_space['num_conv_targets'] = tune.uniform(cfg['study::num_conv_targets_lower'], cfg['study::num_conv_targets_upper']+1)
    if cfg['study::lstm_hidden_size_lower'] != cfg['study::lstm_hidden_size_upper']:
        search_space['lstm_hidden_size'] = tune.uniform(cfg['study::lstm_hidden_size_lower'], cfg['study::lstm_hidden_size_upper']+1)
    if cfg["study::lstm_layers_lower"] != cfg['study::lstm_layers_upper']:
        search_space['num_lstm_layers'] = tune.uniform(cfg["study::lstm_layers_lower"], cfg['study::lstm_layers_upper']+1)

    #Training
    if cfg['study::gradclip_lower'] != cfg['study::gradclip_upper']:
        search_space['gradclip'] = tune.uniform(cfg['study::gradclip_lower'], cfg['study::gradclip_upper'])
    if cfg['study::masking']:
        search_space['use_masking'] = tune.uniform(0, 2)
        if cfg['study::mask_bias_lower'] != cfg['study::mask_bias_upper']:
            search_space['mask_bias'] = tune.quniform(cfg['study::mask_bias_lower'], cfg['study::mask_bias_upper'], 0.1)
    if cfg['study::loss_type']:
        search_space['loss_type'] = tune.uniform(0,2)
    if cfg['study::loss_weight_lower'] != cfg['study::loss_weight_upper']:
        search_space['loss_weight'] = tune.loguniform(cfg['study::loss_weight_lower'], cfg['study::loss_weight_upper'])

    #TAG configuration
    if cfg['study::tag_jumps_lower'] != cfg['study::tag_jumps_upper']:
        search_space['K'] = tune.uniform(cfg['study::tag_jumps_lower'], cfg['study::tag_jumps_upper']+1)

    #GAT and GraphTransformer configuration
    if cfg["study::heads_lower"] != cfg["study::heads_upper"]:
        search_space['heads'] = tune.uniform(cfg["study::heads_lower"], cfg["study::heads_upper"]+1)
    if cfg['study::gat_dropout_lower'] != cfg['study::gat_dropout_upper']:
        search_space['gat_dropout'] = tune.uniform(cfg['study::gat_dropout_lower'], cfg['study::gat_dropout_upper'])

    #Node2Vec configuration
    if cfg['study::embedding_dim_lower'] != cfg['study::embedding_dim_upper']:
        search_space['embedding_dim'] = tune.uniform(cfg['study::embedding_dim_lower'], cfg['study::embedding_dim_upper']+1)
    if cfg['study::walk_length_lower'] != cfg['study::walk_length_upper']:
        search_space['walk_length'] = tune.uniform(cfg['study::walk_length_lower'], cfg['study::walk_length_upper']+1)
    if cfg['study::context_size_lower'] != cfg['study::context_size_upper']:
        search_space['context_size'] = tune.uniform(cfg['study::context_size_lower'], cfg['study::context_size_upper']+1)
    if cfg['study::walks_per_node_lower'] != cfg['study::walks_per_node_upper']:
        search_space['walks_per_node'] = tune.uniform(cfg['study::walks_per_node_lower'], cfg['study::walks_per_node_upper']+1)
    if cfg['study::num_negative_samples_lower'] != cfg['study::num_negative_samples_upper']:
        search_space['num_negative_samples_lower'] = tune.uniform(cfg['study::num_negative_samples_lower'], cfg['study::num_negative_samples_upper']+1)
    if cfg['study::p_lower'] != cfg['study::p_upper']:
        search_space['p'] = tune.loguniform(cfg['study::p_lower'], cfg['study::p_upper'])
    if cfg['study::q_lower'] != cfg['study::q_upper']:
        search_space['q'] = tune.loguniform(cfg['study::q_lower'], cfg['study::q_upper'])

    return search_space

def setup_params_from_search_space(search_space, params, save=False, path=None, ID=None):
    """
    params must already initiated by setup_params which will put the regular values from the cfg file
    setup_params_from_config then overrides the studied values with values from the search_space

    Parameters
    ----------
    search_space : the search_space created by setup_searchspace
    params : the parameters setup by setup_params

    Returns:
    -------
    params

    """
    updated_params = params
    print('Setup params from search space')
    for key in search_space.keys():
        if key in ['LR', 'weight_decay']:
            updated_params[key] = 10**search_space[key]
        elif key == 'reghead_type':
            if search_space[key] < 1:
                updated_params[key] = 'single'
            elif search_space[key] < 2:
                updated_params[key] = 'node_edge'  
            elif search_space[key] < 3:
                updated_params[key] = 'node_node_edge'
        elif key in ['num_layers', 'hidden_size', 'reghead_size', 'reghead_layers', 'num_conv_targets', 'lstm_hidden_size', 'num_lstm_layers', 'K', 'heads']:
            updated_params[key] = int(search_space[key])
        else:
            updated_params[key] = search_space[key]

        
    return updated_params

def setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_seq_len_LDTSF):
    """
    Sets up the parameters dictionary for building and training a model

    Parameters
    ----------
    cfg : preloaded json configuration file

    mask_probs : float array
        probabilities for node masking
    num_features : int
        number of node features in the data
    num_edge_features : int
        number of edge features in the data

    Returns
    -------
    params : dict
        parameter dictionary

    """

    params = {

        'task'  :   cfg['task'],
        'LR' :  cfg['optim::LR'],
        'weight_decay'   :   cfg['optim::weight_decay'],

        "num_features"          :   num_features,
        "num_edge_features"     :   num_edge_features,
        "num_targets"           :   num_targets,

        "num_layers"    :   cfg['num_layers'],
        "hidden_size"   :   cfg['hidden_size'],

        "reghead_size"  :   cfg['reghead_size'],
        "reghead_layers":   cfg['reghead_layers'],
        "reghead_type"  :   cfg['reghead_type'],

        "dropout"       :   cfg["dropout"],

        "use_batchnorm" :   cfg['use_batchnorm'],
        "gradclip"      :   cfg['gradclip'],
        "use_skipcon"   :   cfg['use_skipcon'],
        "use_masking"   :   cfg['use_masking'],
        'mask_bias'     :   cfg['mask_bias'],
        "mask_probs"    :   mask_probs,
        "loss_weight"   :   cfg['weighted_loss_factor'],

        #Params for GAT and GraphTransformer
        "heads"         :   cfg['num_heads'],
        'gat_dropout'   :   cfg['gat_dropout'],

        #Params for TAG
        "K"     :   cfg['tag_jumps'],

        #Params for LSTM
        "num_conv_targets"  :   cfg['num_conv_targets'],
        'lstm_hidden_size'  :   cfg['lstm_hidden_size'],
        'num_lstm_layers'   :   cfg['num_lstm_layers'],
        'max_seq_len_LDTSF'      :   max_seq_len_LDTSF,

        #Params for Node2vec
        'embedding_dim'   :   cfg['embedding_dim'],
        'walk_length'     :   cfg['walk_length'],
        'context_size'    :   cfg['context_size'],
        'walks_per_node'  :   cfg['walks_per_node'],
        'num_negative_samples'    :   cfg['num_negative_samples'],
        'p'     :   cfg['p'],
        'q'     :   cfg['q'],

    }

    return params

def save_output(output, labels, test_output, test_labels, name=""):
    with open("results/" + "output"+name+".pt", "wb") as f:
        torch.save(output, f)
    with open("results/" + "labels"+name+".pt", "wb") as f:
        torch.save(labels, f)
    with open("results/" + "test_output"+name+".pt", "wb") as f:
        torch.save(test_output, f)
    with open("results/" + "test_labels"+name+".pt", "wb") as f:
        torch.save(test_labels, f)


def choose_criterion(task, weighted_loss_label, weighted_loss_factor, cfg, device):
            # Init Criterion
        if task in ['GraphClass', 'typeIIClass']:
            criterion = torch.nn.CrossEntropyLoss()
        elif weighted_loss_label:
            criterion = weighted_loss_label(
            factor=torch.tensor(weighted_loss_factor))
        elif task == 'StateReg':
            criterion = state_loss(weighted_loss_factor)
        elif task == 'StateRegPI':
            criterion = state_loss_power_injection(cfg, weighted_loss_factor, cfg['PI_factor'], device)
        elif task == 'NodeReg':
            criterion = torch.nn.MSELoss(reduction='mean')
        return criterion

def setup_datasets_and_loaders(cfg, N_CPUS, pin_memory):
    max_seq_len_LDTSF = -1
    if cfg['model'] == 'Node2Vec':
         trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
         trainloader, testloader, max_seq_len_LDTSF = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
    elif 'LSTM' in cfg['model']:
            # Split dataset into train and test indices
        trainset, testset = create_lstm_datasets(cfg["dataset::path"], cfg['train_size'], cfg['manual_seed'], 
                                                     stormsplit=cfg['stormsplit'], max_seq_len=cfg['max_seq_length'], autoregressive=cfg['autoregressive'])
            # Create DataLoaders for train and test sets
        trainloader = create_lstm_dataloader(trainset, batch_size=cfg['train_set::batchsize'], shuffle=True, pin_memory=pin_memory, num_workers=N_CPUS)
        testloader = create_lstm_dataloader(testset, batch_size=cfg['test_set::batchsize'], shuffle=False, pin_memory=pin_memory, num_workers=N_CPUS)
    else:
         trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
         trainloader, testloader, max_seq_len_LDTSF = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'], task=cfg['task'])
    return max_seq_len_LDTSF, trainset, trainloader, testloader


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim > 0 else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_params(path, params, ID=None):
    del params['mask_probs']
    with open(os.path.join(path,'results/', ID+'_params.json'), 'w') as f:
        json.dump(params, f, default=tensor_to_serializable, indent=4)

def multiclass_classification(output, labels, N_bins):

    output = output.reshape(-1)
    labels = labels.reshape(-1)
    N_nodes = len(output)
    labelclasses = torch.zeros(N_bins+1)
    outputclasses = torch.zeros(N_bins+1)
    matrix = torch.zeros([N_bins+1, N_bins+1])
    for node in range(N_nodes):
        for i in range(N_bins+1):
            if output[node] <= (1/N_bins)*i:
                outputclasses[i] += 1
                break;
        for j in range(N_bins+1):
            if labels[node] <= (1/N_bins)*j:
                labelclasses[j] += 1
                matrix[j,i] += 1
                break;
    return outputclasses, labelclasses, matrix

class state_loss(torch.nn.Module):
    def __init__(self, edge_factor):
        super(state_loss, self).__init__()
        self.edge_factor = edge_factor
        self.node_loss = torch.nn.MSELoss(reduction='mean')
        self.edge_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        print(f'Using State Loss with edge factor {self.edge_factor}')

    def forward(self, node_output, edge_output, node_labels, edge_labels):
        node_loss = self.node_loss(node_output, node_labels)
        edge_loss = self.edge_loss(edge_output, edge_labels.reshape(-1))
        loss = node_loss + edge_loss*self.edge_factor
        return loss, node_loss, edge_loss
    
class state_loss_power_injection(torch.nn.Module):
    def __init__(self, cfg, edge_factor, PI_factor, device):
        """
        Initializes the state loss with an edge factor and provides the necessary static information for the power injection loss.
        Parameters
        ----------
        edge_factor : float
            Factor to scale the edge loss.
        b_edge_attr : torch.Tensor, optional
            Edge attributes representing the shunt admittance (jb/2) for each edge.
        Y_raw : torch.Tensor
            Raw admittance matrix of the grid, used to build the Y matrix based on the edge predictions. Used to calculate the power injections.
        basekV : torch.Tensor
            Base voltage of the grid, used to calculate the power injections.
        min_max : dict
            Dictionary containing the min and max values for denormalization of the predictions.
        """

        super(state_loss_power_injection, self).__init__()
        self.edge_factor = edge_factor
        self.PI_factor = PI_factor
        self.node_loss = torch.nn.MSELoss(reduction='mean')
        self.edge_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.PI_loss = torch.nn.MSELoss(reduction='mean')

        self.device=device
        #Things needed for the power injection loss
        static_data = torch.load(os.path.join(cfg['dataset::path'], 'processed/data_static.pt'))    #Used for pytorch branch IDs of fully functioning grid
        pwsdata = scipy.io.loadmat(os.path.join(cfg['dataset::path'], 'raw/pwsdata.mat'))    #  
        self.edge_index = static_data.edge_index
        self.Y_raw = torch.tensor(pwsdata['clusterresult_'][0,0][10] )
        self.Y_raw = torch.complex(torch.tensor(self.Y_raw.real), torch.tensor(self.Y_raw.imag)).type(torch.complex64).to(self.device)

     
        bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
        branch_data = pwsdata['clusterresult_'][0,0][4]
        b = torch.tensor(branch_data[:,4] )
        self.b_edge_attr = self.create_b_edge_attr(bus_IDs, branch_data, self.edge_index, b).to(self.device)

        print(f'Using State Loss with edge factor {self.edge_factor} and PI factor {self.PI_factor}')

    def forward(self, node_output, edge_output, node_labels, edge_labels):
        #denormalized_output, denormalized_labels = self.denormalize((node_output, edge_output), (node_labels, edge_labels))
        node_loss = self.node_loss(node_output, node_labels)

        edge_loss = self.edge_loss(edge_output, edge_labels.reshape(-1))

        S = self.calculate_S((node_output, edge_output), (node_labels, edge_labels), use_edge_labels=True)
        S_true = self.calculate_S((node_labels, edge_labels), (node_labels, edge_labels), use_edge_labels=True)
        PI_loss = self.PI_loss(torch.view_as_real(S), torch.view_as_real(S_true))

        loss = node_loss + self.edge_factor*edge_loss + self.PI_factor*PI_loss
        return loss, node_loss, edge_loss, PI_loss
    
    
    def build_Y_matrix_from_predictions(self, edge_predictions):
        Y = self.Y_raw.clone()
        Y = Y.to(torch.complex64)
        inactive_edges = torch.where(edge_predictions.flatten() == 0)[0]

        for idx in inactive_edges:
            #print('inactive_edges: ', len(inactive_edges))
            #print(idx)
            i, j = self.edge_index[:,idx]
            y_ij = Y[i, j]
    
            if i!=j:
                Y[i, i] += y_ij - self.b_edge_attr[idx]
                Y[i, j] = 0
            """if i < 10:
                print('iterative')
                print(i,j)
                print(Y[i,j])

        Y = self.Y_raw.clone()
        Y = Y.to(torch.complex64)
        
        mask = (edge_predictions == 0)  # inactive edges
        i_idx, j_idx = self.edge_index[:, mask]

        # zero out Y[i, j] for inactive edges
        Y[i_idx, j_idx] = 0

        # fix diagonal in a vectorized way
        diag_add = Y[i_idx, j_idx] - self.b_edge_attr[mask]
        Y[i_idx, i_idx] += diag_add
        print('vectorized')
        print(Y[:10, :10])"""


        Y[abs(Y.real)<0.001] = 0j
        return Y #torch.complex(torch.tensor(Y.real), torch.tensor(Y.imag)).type(torch.complex64).to(self.device) 
    
    def calculate_S(self, output, labels, use_edge_labels):


        S_all = []


        for i in range(int(len(output[0])/2000)):
            if use_edge_labels:
                Y_instance = self.build_Y_matrix_from_predictions(labels[1][i])
            else:
                instance_output = (output[0], torch.nn.functional.gumbel_softmax(output[1][i*7064:(i+1)*7064], tau=1.0, hard=True, dim=1))  # shape: [batch_size]
                Y_instance = self.build_Y_matrix_from_predictions(instance_output[1][:,0])

            V = output[0][i*2000:(i+1)*2000,:].float()
            V = torch.complex(V[:, 0], V[:, 1])

            YV= Y_instance.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
            S = V * YV.conj()

            S_all.append(S)

        return torch.stack(S_all)
    
    def create_b_edge_attr(self, bus_IDs, branch_data, edge_index, b):
        bus_id_map = {int(bus_id): idx for idx, bus_id in enumerate(bus_IDs)}
        from_buses_raw = branch_data[:, 0].astype(int)
        to_buses_raw   = branch_data[:, 1].astype(int)

        line_to_shunt = {}
        for fb_raw, tb_raw, b in zip(from_buses_raw, to_buses_raw, b):
            fb = bus_id_map[fb_raw]
            tb = bus_id_map[tb_raw]
            i, j = sorted((fb, tb))
            line_to_shunt[(i, j)] = 1j * b / 2  # jb/2

    # Step 4: Assign jb/2 to each edge in edge_index
        shunt_attr = torch.zeros(edge_index.shape[1], dtype=torch.cfloat)
        for k in range(edge_index.shape[1]):
            i = edge_index[0, k].item()
            j = edge_index[1, k].item()
            key = tuple(sorted((i, j)))
            if key in line_to_shunt:
                shunt_attr[k] = torch.tensor(line_to_shunt[key], dtype=torch.cfloat)
        return shunt_attr


class weighted_loss_label(torch.nn.Module):
    """
    weights the loss with a constant factor depending on wether the label is >0 or not
    """
    def __init__(self, factor):
        super(weighted_loss_label, self).__init__()
        self.factor = torch.sqrt(factor)
        self.node_loss= torch.nn.MSELoss(reduction='mean')
        self.edge_loss

    def forward(self, output, labels):
        print('Using weighted loss label')
        output_ = output.clone()
        labels_ = labels.clone()
        output_[labels>0] = output_[labels>0]*self.factor
        labels_[labels>0] = labels_[labels>0].clone()*self.factor
        return self.baseloss(self.factor*output_,self.factor*labels_)


class weighted_loss_var(torch.nn.Module):
    """
    weights the loss depending on the label variance at each node
    """
    def __init__(self, var, device):
        super(weighted_loss_var, self).__init__()
        self.weights = torch.sqrt(var).to(device)
        self.baseloss = torch.nn.MSELoss(reduction='mean').to(device)

    def forward(self, output ,labels):
        output_ = output.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        labels_ = labels.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        return self.baseloss(output_.reshape(-1), labels_.reshape(-1))
    

def create_data_from_prediction(predicted_node_features, predicted_edge_status, reference, node_labels, edge_labels):
    """
    Create a new Data object from predicted node features and edge status.
    This function assumes that the reference Data object is the static data (i.e. 7064 edges)
    and that the edge status is a binary tensor indicating the presence of edges.
    """

    # Mask for active lines
    active_mask = (predicted_edge_status == 0)  # [E], boolean

    # Apply mask to get filtered topology and admittances
    updated_edge_index = reference.edge_index[:, active_mask]       # [2, E']
    updated_edge_attr  = reference.edge_attr[active_mask]           # [E']


    # Get source and target nodes of each edge
    src, dst = updated_edge_index  # [E], [E]

    V = predicted_node_features[:,0] + 1j * predicted_node_features[:,1]  # [E], complex tensor
    # Get edge admittances Y_ij
    #Y_ij = updated_edge_attr       # [E], complex tensor
    Y_ij = updated_edge_attr[:, 0] + 1j * updated_edge_attr[:, 1]  # shape [E], dtype=torch.complex64 or complex128


    # V_j values at source nodes (i.e., neighbor voltages)
    V_j = V[src]           # [E]
    print('V_j:', V_j.shape)
    print('Y_ij:', Y_ij.shape)
    # Compute message: Y_ij * V_j
    messages = Y_ij * V_j  # [E]


    # Aggregate incoming messages at target node (i.e., YV at each node)
    YV = scatter_add(messages, dst, dim=0, dim_size=V.shape[0])  # [N]
    S = V * YV.conj()  # [N]
    new_node_features = torch.stack((predicted_node_features[:,0], predicted_node_features[:,1], S.real, S.imag), dim=1)  # [N, 2]


    new_data = Data(
        x=new_node_features,  # shape [num_nodes, num_node_features]
        edge_index = updated_edge_index,
        edge_attr = updated_edge_attr,
        node_labels = node_labels,  # shape [num_nodes, num_node_features]
        edge_labels = edge_labels,  # shape [num_edges, num_edge_features]
        # optionally include dummy y or node_labels if needed
    )
    return new_data





