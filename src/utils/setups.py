from ray import tune
from datasets.dataset import create_datasets, create_loaders
from datasets.dataset_graphlstm import create_lstm_datasets, create_lstm_dataloader


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


def setup_datasets_and_loaders(cfg, N_CPUS, pin_memory):
    max_seq_len_LDTSF = -1
    PROCESSING_LSTM_DATA = False
    if cfg['model'] == 'Node2Vec':
         trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
         trainloader, testloader, max_seq_len_LDTSF = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
    elif 'LSTM' in cfg['model']:
            # Split dataset into train and test indices
        trainset, testset = create_lstm_datasets(cfg)
            # Create DataLoaders for train and test sets
        trainloader = create_lstm_dataloader(trainset, batch_size=cfg['train_set::batchsize'], shuffle=True, pin_memory=pin_memory, num_workers=N_CPUS)
        testloader = create_lstm_dataloader(testset, batch_size=cfg['test_set::batchsize'], shuffle=False, pin_memory=pin_memory, num_workers=N_CPUS)
    else:
        trainset, testset, PROCESSING_LSTM_DATA = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
        trainloader, testloader, max_seq_len_LDTSF = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'], task=cfg['task'])

    return max_seq_len_LDTSF, trainset, trainloader, testloader, PROCESSING_LSTM_DATA

