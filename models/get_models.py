from models.tag import TAGNodeReg
from models.gat import GAT
from models.gine import GINE
from models.baselines import ridge, MLP



def get_model(cfg, params):
    """
    Helper function which gets the desired model and
    initializes it with the parameters specified in
    params
    """
    if cfg['task'] == 'NodeReg':
        #RIDGE
        if cfg['model'] == 'Ridge':
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
                hidden_size     = params["hidden_size"],
                num_layers      = params["num_layers"],
                dropout         = params['dropout'],
                K               = params['K'],
                reghead_size    = params['reghead_size'],
                reghead_layers  = params['reghead_layers'],
                use_skipcon     = params['use_skipcon'],
                use_batchnorm   = params['use_batchnorm']
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
        
        #Other (mainly GINE)
        else:
            print('Using GINE!\n')
            try:
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
                )
            except NameError:
                raise NameError("Unknown model selected. Change model in gnn/configuration.json")
                
                
    elif cfg['task'] == 'GraphReg':
        if cfg['model'] == 'TAG':
            print('Using TAGGraphReg!\n')
            model = TAGGraphReg(
                hidden_size     = params["hidden_size"],
                num_layers      = params["num_layers"],
                dropout         = params['dropout'],
                K               = params['K'],
                reghead_size    = params['reghead_size'],
                reghead_layers  = params['reghead_layers'],
                use_skipcon     = params['use_skipcon'],
                use_batchnorm   = params['use_batchnorm']
                )
            
            #GAT
        elif cfg['model'] == 'GAT':
            print('Using GAT!\n')
            model = GATGraphReg(
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
        elif cfg['model'] == 'GINE':
            print('Using GINE!\n')
            try:
                model = GINEGraphReg(
                    num_node_features   = params["num_features"],
                    num_edge_features   = params["num_edge_features"],
                    num_targets     = params["num_targets"],
                    hidden_size     = params["hidden_size"],
                    num_layers      = params["num_layers"],
                    dropout         = params['dropout'],
                    use_skipcon     = params['use_skipcon'],
                    reghead_size    = params['reghead_size'],
                    reghead_layers  = params['reghead_layers'],
                )
            except NameError:
                raise NameError("Unknown model selected. Change model in gnn/configuration.json")
                
        elif cfg['model'] == 'Ridge':
            print('Using RIDGE!\n')
            model = ridge_graphreg(
                num_node_features = params["num_features"],
                hidden_size = params["hidden_size"],
                )
        
        #Multilayer Perceptron
        elif cfg['model'] == 'MLP' or cfg['model'] == 'Node2Vec':
            print('Using MLP or Node2Vec with MLP!\n')
            model = MLP_graphreg(
                num_node_features   = params['num_features'],
                hidden_size         = params['hidden_size'],
                num_layers      =   params['num_layers'],
                dropout         =   params['dropout'],
                use_skipcon     = params['use_skipcon'],
                use_batchnorm   = params['use_batchnorm']
                )
            
    else:
        assert False, 'Only NodeReg or GraphReg allowed as tasks!'


    return model.float()