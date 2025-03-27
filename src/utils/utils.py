from typing import List, Optional, Union

import torch
import json

import torch.nn
import numpy as np
import matplotlib.pyplot as plt
import os
from ray import tune
import warnings


def check_config_conflicts(cfg):
    assert not (cfg['crossvalidation'] and cfg['study::run']),  'can only run a study or the crossvalidation not both'
    assert not (cfg['data'] == 'DC' and cfg['stormsplit']>0),   'Stormsplit can only be used with AC data'
    assert not (cfg['edge_attr'] == 'multi' and cfg['model'] == 'TAG'), 'TAG can only be used with Y as edge_attr not with multi'
    assert not (cfg['data'] == 'LDTSF' and cfg['task'] == 'NodeReg'),   'LDTSF Only works with GraphReg and GraphClass'
    assert not (cfg['data'] == 'LDTSF' and cfg['model'] != 'lstm'),     'LDTSF Only works with lstm as model'
    assert not (cfg['data'] == 'AC' and cfg['task'] == 'GraphClass'),   'None of the models working with AC data has GraphClass implemented' 
    if not (cfg['data'] == 'LSTM' and cfg['model'] == 'GATLSTM'): 
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
                updated_params[key] = 'double'  
            elif search_space[key] < 3:
                updated_params[key] = 'triple'
        else:
            updated_params[key] = search_space[key]

        
    return updated_params

def setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_length, save=False):
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
        'max_seq_length'      :   max_length,

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
        edge_loss = self.edge_loss(edge_output, edge_labels.reshape(-1))*self.edge_factor
        loss = node_loss + edge_loss
        return loss, node_loss, edge_loss


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




