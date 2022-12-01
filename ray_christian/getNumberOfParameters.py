import torch as torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


from torch_geometric.data import DataLoader
from gnn_models import GNNmodule, gnn_snbs_surv, ArmaNet_ray, GCNNet_ray, SAGENet_ray, TAGNet_ray
from torch_geometric.nn import Node2Vec

from pathlib import Path

from utils import gnn_model_summary


cfg = {}
config = cfg


# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 2
# cfg["num_channels"] = [1, 200, 1]
# cfg["batch_norm_index"] = [True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [8, 8]
# cfg["ARMA::num_internal_stacks"] = [100, 100]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True
# # GCN
# cfg["GCN::improved"] = True
# # TAG
# cfg["TAG::K_hops"] = [3, 3]


# # # #############################################################################################
# # ARMANet1
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 1
# cfg["num_channels"] = [1, 1]
# cfg["batch_norm_index"] = [True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [12]
# cfg["ARMA::num_internal_stacks"] = [800000]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True
# # ##############################################################################################

# #############################################################################################
# # ARMANet2
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 2
# cfg["num_channels"] = [1, 200, 1]
# cfg["batch_norm_index"] = [True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [8, 8]
# cfg["ARMA::num_internal_stacks"] = [100, 100]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True


# # ARMANet3
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 3
# cfg["num_channels"] = [1, 120, 120, 1]
# cfg["batch_norm_index"] = [True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10]
# cfg["ARMA::num_internal_stacks"] = [70, 70, 70]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True

# # ARMANet4
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 4
# cfg["num_channels"] = [1, 100, 100, 100,1]
# cfg["batch_norm_index"] = [True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10,10]
# cfg["ARMA::num_internal_stacks"] = [60, 60, 60, 60]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True

# # ARMANet5
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 5
# cfg["num_channels"] = [1, 90, 90, 90, 90, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10,10, 10]
# cfg["ARMA::num_internal_stacks"] = [55, 55, 55, 55, 55]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True

# # ARMANet6
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 6
# cfg["num_channels"] = [1, 80, 80, 80, 80, 80, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10,10, 10, 10]
# cfg["ARMA::num_internal_stacks"] = [53, 53, 53, 53, 53, 53]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True


# # ARMANet7_idea, but wrong setting
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 7
# cfg["num_channels"] = [1, 78, 78, 78, 78, 78, 78,1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10, 10, 10, 10, 10]
# cfg["ARMA::num_internal_stacks"] = [44, 44, 44, 44, 44, 44, 44]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True


# # ARMANet7
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 7
# cfg["num_channels"] = [1, 78, 78, 78, 78, 78, 78,1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10, 10, 10, 10, 10]
# cfg["ARMA::num_internal_stacks"] = [53, 53, 53, 53, 53, 53, 53]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True

# # ARMANet8
# # model settings
# cfg["model_name"] = "ArmaNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 8
# cfg["num_channels"] = [1, 74, 74, 74, 74, 74, 74, 74, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True]
# # ARMA
# cfg["ARMA::num_internal_layers"] = [10, 10, 10, 10, 10, 10, 10, 10]
# cfg["ARMA::num_internal_stacks"] = [42, 42, 42, 42, 42, 42, 42, 42]
# cfg["ARMA::dropout"] = .25
# cfg["ARMA::shared_weights"] = True

# ##############################################################################################

# #############################################################################################
# # GCNNet3
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 3
# cfg["num_channels"] = [1, 2000, 2000, 1]
# cfg["batch_norm_index"] = [True, True, True]
# cfg["GCN::improved"] = True

# # GCNNet4
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 4
# cfg["num_channels"] = [1, 1400, 1400, 1400, 1]
# cfg["batch_norm_index"] = [True, True, True, True]
# cfg["GCN::improved"] = True

# # GCNNet5
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 5
# cfg["num_channels"] = [1, 1170, 1170, 1170, 1170, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True]
# cfg["GCN::improved"] = True


# # GCNNet6
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 6
# cfg["num_channels"] = [1, 1000, 1000, 1000, 1000, 1000, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True]
# cfg["GCN::improved"] = True

# # GCNNet7
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 7
# cfg["num_channels"] = [1, 900, 900, 900, 900, 900, 900, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True]
# cfg["GCN::improved"] = True

# # GCNNet8
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 8
# cfg["num_channels"] = [1, 840, 840, 840, 840, 840, 840, 840, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True]
# cfg["GCN::improved"] = True

# # GCNNet9
# # model settings
# cfg["model_name"] = "GCNNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 9
# cfg["num_channels"] = [1, 780, 780, 780, 780, 780, 780, 780, 780, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True, True]
# cfg["GCN::improved"] = True

# # ##############################################################################################

# # # SAGENet3
# # model settings
# cfg["model_name"] = "SAGENet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 3
# cfg["num_channels"] = [1, 1500, 1500, 1]
# cfg["batch_norm_index"] = [True, True, True]


# # # TAGNet2
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 2
# cfg["num_channels"] = [1, 100000, 1]
# cfg["batch_norm_index"] = [True, True, True]
# cfg["TAG::K_hops"] = [12, 12]

# # # TAGNet3
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 3
# cfg["num_channels"] = [1, 580, 580, 1]
# cfg["batch_norm_index"] = [True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12]

# # # TAGNet4
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 4
# cfg["num_channels"] = [1, 395, 395, 395, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12]

# # # TAGNet5
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 5
# cfg["num_channels"] = [1, 350, 350, 350, 350, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12, 12]


# # # TAGNet6
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 6
# cfg["num_channels"] = [1, 280, 280, 280, 280, 280, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12, 12, 12]

# # # TAGNet7
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 7
# cfg["num_channels"] = [1, 250, 250, 250, 250, 250, 250, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12, 12, 12, 12]

# # # TAGNet8
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 8
# cfg["num_channels"] = [1, 235, 235, 235, 235, 235, 235, 235, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12, 12, 12, 12, 12]

# # # # TAGNet9
# # model settings
# cfg["model_name"] = "TAGNet_ray"
# cfg["final_linear_layer"] = False
# cfg["num_layers"] = 9
# cfg["num_channels"] = [1, 215, 215, 215, 215, 215, 215, 215, 215, 1]
# cfg["batch_norm_index"] = [True, True, True, True, True, True, True, True, True, True]
# cfg["TAG::K_hops"] = [12, 12, 12, 12, 12, 12, 12, 12, 12]

if config["model_name"] == "ArmaNet_bench":
    model = ArmaNet_bench()
elif config["model_name"] == "ArmaNet_ray":
    model = ArmaNet_ray(num_layers=config["num_layers"], num_channels=config["num_channels"],
                        num_internal_layers=config["ARMA::num_internal_layers"], num_internal_stacks=config["ARMA::num_internal_stacks"], batch_norm_index=config["batch_norm_index"], shared_weights=config["ARMA::shared_weights"], dropout=config["ARMA::dropout"], final_linear_layer=config["final_linear_layer"])
elif config["model_name"] == "GCNNet_bench":
    model = GCNNet_bench()
elif config["model_name"] == "GCNNet_ray":
    model = GCNNet_ray(num_layers=config["num_layers"], num_channels=config["num_channels"], improved=config["GCN::improved"],
                        batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
elif config["model_name"] == "SAGENet_ray":
    model = SAGENet_ray(num_layers=config["num_layers"], num_channels=config["num_channels"],
                        batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
elif config["model_name"] == "TAGNet_bench":
    model = TAGNet_bench()
elif config["model_name"] == "TAGNet_ray":
    model = TAGNet_ray(num_layers=config["num_layers"], num_channels=config["num_channels"], K_hops=config["TAG::K_hops"],
                        batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
else:
    print("error: model type unkown")


gnn_model_summary(model)

print("script finished")