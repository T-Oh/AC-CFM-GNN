from models.tag import TAGNodeReg
from models.gat import GAT
from models.gine import GINE
from models.baselines import ridge, MLP
from models.gtrans import GraphTransformer
from models.lstm import GCNLSTM
from models.lstm_LDTSF import LSTM_LDTSF
from models.gatlstm import GAT_LSTM
from models.mlplstm import LSTM_Baseline
from models.taglstm import TAG_LSTM




def get_model(cfg, params):
    """
    Helper function which gets the desired model and
    initializes it with the parameters specified in
    params
    """


    if 'LDTSF' in cfg['data']:
        if cfg['model'] == 'lstm':
            model = LSTM_LDTSF(
                num_features        = params['num_features'],
                num_targets         = params['num_targets'],
                lstm_hidden_size    = params['lstm_hidden_size'], 
                num_lstm_layers     = params['num_lstm_layers'],
                reghead_size        = params['reghead_size'],
                reghead_layers      = params['reghead_layers'],
                gat_dropout         = params['gat_dropout'],
                max_seq_length      = params['max_seq_len_LDTSF'],
                task                = params['task']   
            )
        else:
            print('WARNING: DATATYPE SET TO LDTSF BUT MODEL IS NOT LSTM - CONTINUEING WITH LSTM_LDTSF AS MODEL')
    #RIDGE
    elif cfg['model'] == 'Ridge':
        print('Using RIDGE!\n')
        model = ridge(
            num_node_features = params["num_features"],
            hidden_size = params["hidden_size"],
            )

    #Multilayer Perceptron
    elif cfg['model'] == 'MLP' or cfg['model'] == 'Node2Vec':
        print('Using MLP or Node2Vec with MLP!\n')
        model = MLP(
            num_node_features   = params['num_features'],
            hidden_size         = params['hidden_size'],
            num_layers      =   params['num_layers'],
            dropout         =   params['dropout'],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm']
            )

    #TAG
    elif cfg['model'] == 'TAG':
        print('Using TAG!\n')
        model = TAGNodeReg(
            num_node_features   = params['num_features'],
            num_targets     = params['num_targets'],
            hidden_size     = params["hidden_size"],
            num_layers      = params["num_layers"],
            dropout         = params['dropout'],
            K               = params['K'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm'],
            task            = params['task']
            )

    #GAT
    elif cfg['model'] == 'GAT':
        print('Using GAT!\n')
        model = GAT(
            num_node_features = params['num_features'],
            num_edge_features   = params["num_edge_features"],
            num_targets     = params["num_targets"],
            hidden_size     = params["hidden_size"],
            num_layers      = params["num_layers"],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            dropout         = params['dropout'],
            gat_dropout     = params['gat_dropout'],
            num_heads       = params["heads"],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm']
            )

    #GINE
    elif cfg['model'] == 'GINE':
        print('Using GINE!\n')
        model = eval(cfg["model"])(
            num_node_features   = params["num_features"],
            num_edge_features   = params["num_edge_features"],
            num_targets     = params["num_targets"],
            hidden_size     = params["hidden_size"],
            num_layers      = params["num_layers"],
            dropout         = params['dropout'],
            use_skipcon     = params['use_skipcon'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            task            = params['task']
        )
    #GraphTransformer
    elif cfg['model'] == 'GTrans':
        print('Using Graph Transformer!\n')
        model = GraphTransformer(
            num_node_features   = params["num_features"],
            num_edge_features   = params["num_edge_features"],
            num_targets     = params["num_targets"],
            hidden_size     = params["hidden_size"],
            num_layers      = params["num_layers"],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            dropout         = params['dropout'],
            gat_dropout     = params['gat_dropout'],
            num_heads       = params["heads"],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm'],
            checkpoint      = cfg['checkpointing'],
            task            = params['task']
        )
    elif cfg['model'] == 'LSTM':
        print('Using GCN LSTM')
        model = GCNLSTM(
            num_node_features   = params["num_features"],
            conv_hidden_size    = params['hidden_size'], 
            num_conv_targets    = params["num_conv_targets"],  
            num_conv_layers     = params['num_layers'],
            lstm_hidden_size    = params['lstm_hidden_size'],
            num_lstm_layers     = params['num_lstm_layers'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            num_targets     = params['num_targets'],
            dropout         = params['dropout'],
            gat_dropout     = params['gat_dropout'],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm'],
            len_sequence    = params['len_sequence']
        )
    elif cfg['model'] == 'GATLSTM':
        print('Using GCN LSTM')
        model = GAT_LSTM(
            num_node_features   = params["num_features"],
            conv_hidden_size    = params['hidden_size'], 
            num_conv_targets    = params["num_conv_targets"],  
            num_conv_layers     = params['num_layers'],
            lstm_hidden_size    = params['lstm_hidden_size'],
            num_lstm_layers     = params['num_lstm_layers'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            dropout         = params['dropout'],
            gat_dropout     = params['gat_dropout'],
            num_heads       = params["heads"],  
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm'],
            task                = params['task']   
        )

    elif cfg['model'] == 'TAGLSTM':
        print('Using TAG LSTM')
        model = TAG_LSTM(
            num_node_features   = params["num_features"],
            conv_hidden_size    = params['hidden_size'], 
            num_conv_targets    = params["num_conv_targets"],  
            num_conv_layers     = params['num_layers'],
            lstm_hidden_size    = params['lstm_hidden_size'],
            num_lstm_layers     = params['num_lstm_layers'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            dropout         = params['dropout'],
            K               = params['K'],
            use_skipcon     = params['use_skipcon'],
            use_batchnorm   = params['use_batchnorm'],
            task                = params['task']   
        )

    elif cfg['model'] == 'MLPLSTM':
        print('Using Linear LSTM')
        model = LSTM_Baseline(
            num_node_features   = params["num_features"],
            lstm_hidden_size    = params['lstm_hidden_size'],
            num_lstm_layers     = params['num_lstm_layers'],
            reghead_size    = params['reghead_size'],
            reghead_layers  = params['reghead_layers'],
            dropout         = params['dropout'],
            task                = params['task']   
        )


    
    else: 
        raise NameError("Unknown model selected. Change model in gnn/configuration.json")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('NUMBER OF PARAMETERS:')
    print(pytorch_total_params)



    return model.float()
