from models.arma import ArmaNet_ray, make_list_Arma_internal_layers, make_list_Arma_internal_stacks, make_list_number_of_channels, ArmaConvModule
from models.tag import TAG, TAGNet01, TAGTest, TAGNodeReg
from models.gcn import GCN
from models.gat import GAT
from models.sage import SAGE
from models.gine import GINE
from models.baselines import ridge, node2vec


def get_model(cfg, params):
    """
    Helper function which gets the desired model and
    initializes it with the parameters specified in
    params
    """
    if cfg["model"] == "ArmaConvModule":
            num_internal_layers = 1
            num_internal_stacks = 1
            num_channels = make_list_number_of_channels(cfg)
            model = ArmaConvModule(num_channels_in=cfg["num_channels1"],num_channels_out=cfg["num_channels2"], activation=cfg["activation"],
                                num_internal_layers=num_internal_layers, num_internal_stacks=num_internal_stacks, shared_weights=cfg["ARMA::shared_weights"], dropout=cfg["ARMA::dropout"])

    elif cfg["model"] == "ArmaConvModule":
            num_internal_layers = make_list_Arma_internal_layers(cfg)
            num_internal_stacks = make_list_Arma_internal_stacks(cfg)
            num_channels = make_list_number_of_channels(cfg)
            model = ArmaConvModule(num_layers=cfg["num_layers"], num_channels=num_channels, activation=cfg["activation"],
                                num_internal_layers=num_internal_layers, num_internal_stacks=num_internal_stacks, batch_norm_index=cfg["batch_norm_index"], shared_weights=cfg["ARMA::shared_weights"], dropout=cfg["ARMA::dropout"], final_linear_layer=cfg["final_linear_layer"])
            
    elif cfg['model'] == 'Node2Vec':
        model = node2vec(
            edge_index      = params['edge_index'],
            embedding_dim   = params['embedding_dim'],
            walk_length     = params['walk_length'],
            context_size    = params['context_size'],
            walks_per_node  = params['walks_per_node'])
        
    elif cfg['model'] == 'Ridge':
        model = ridge(
            num_node_features = params["num_features"],
            hidden_size = params["hidden_size"],
            )

    else:
        try:
            model = eval(cfg["model"])(
                num_node_features   = params["num_features"],
                num_edge_features   = params["num_edge_features"],
                num_targets     = params["num_targets"],
                hidden_size     = params["hidden_size"],
                num_layers      = params["num_layers"],
                dropout         = params['dropout'],
                dropout_temp    = params['dropout_temp'],
                num_heads       = params["heads"],
                use_batchnorm   = params['use_batchnorm'],
                use_skipcon     = params['use_skipcon'],
                reghead_size    = params['reghead_size'],
                reghead_layers  = params['reghead_layers'],
                use_masking     = params['use_masking'],
                mask_probs      =params['mask_probs']
            )
        except NameError:
            raise NameError("Unknown model selected. Change model in gnn/configuration.json")


    return model.float()