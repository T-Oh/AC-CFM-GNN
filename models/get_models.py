from models.arma import ArmaNet_ray, make_list_Arma_internal_layers, make_list_Arma_internal_stacks, make_list_number_of_channels
from models.tag import TAG, TAGNet01, TAGTest
from models.gcn import GCN


def get_model(cfg, params, num_features, num_targets):
    """
    Helper function which gets the desired model and
    initializes it with the parameters specified in
    params
    """
    if cfg["model"] == "ArmaNet_ray":
            num_internal_layers = make_list_Arma_internal_layers(cfg)
            num_internal_stacks = make_list_Arma_internal_stacks(cfg)
            num_channels = make_list_number_of_channels(cfg)
            model = ArmaNet_ray(num_layers=cfg["num_layers"], num_channels=num_channels, activation=cfg["activation"],
                                num_internal_layers=num_internal_layers, num_internal_stacks=num_internal_stacks, batch_norm_index=cfg["batch_norm_index"], shared_weights=cfg["ARMA::shared_weights"], dropout=cfg["ARMA::dropout"], final_linear_layer=cfg["final_linear_layer"])
    else:
        try:
            model = eval(cfg["model"])(
                num_node_features=num_features,
                num_targets=num_targets,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"]
            )
        except NameError:
            raise NameError("Unknown model selected. Change model in gnn/configuration.json")


    return model.float()