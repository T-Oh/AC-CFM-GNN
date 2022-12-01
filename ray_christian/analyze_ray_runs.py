import pandas as pd
import numpy as np
import os as os
from pathlib import Path
import json
import torch as torch

import matplotlib.pyplot as plt

from gnn_models import GNNmodule, gnn_snbs_surv, ArmaNet_ray, GCNNet_ray, SAGENet_ray, TAGNet_ray



result_keys = ["test_acc", "test_R2"]

result_modes = {}
result_modes["train_loss"] = "min"
result_modes["valid_loss"] = "min"
result_modes["test_loss"] = "min"

result_modes["train_acc"] = "max"
result_modes["valid_acc"] = "max"
result_modes["test_acc"] = "max"

result_modes["train_R2"] = "max"
result_modes["valid_R2"] = "max"
result_modes["test_R2"] = "max"

def get_result_value(result, mode):
    if mode == "min":
        return min(result)
    if mode == "max":
        return max(result)

def read_input_data(path_runs,numRuns,parameter_keys, necessary_parameters_per_model):
    listDirectories = []
    for file in sorted(os.listdir(path_runs)):
        if "NN_tune_trainable" in file:
            listDirectories.append(file)
    results_dataframe = pd.DataFrame(
        columns=parameter_keys, index=np.arange(numRuns))

    for i in range(numRuns):
        parameter_file_name = Path(path_runs + '/' + listDirectories[i] + '/params.json')
        with open(parameter_file_name) as f:
            parameter_json = json.load(f)
        for j in range(len(parameter_keys)):
            vName = parameter_keys[j]
            parameters_loaded = parameter_json[vName]
            if type(parameters_loaded) == list:
                num_parameters_loades = len(parameters_loaded)
                for k in range(num_parameters_loades):
                    name_dataframe = vName + '_' + str(k)
                    results_dataframe.at[i, name_dataframe] = parameters_loaded[k]        
            else:
                results_dataframe.at[i, vName] = parameter_json[vName]
        result_file_name = Path(
            path_runs + '/' + listDirectories[i] + '/progress.csv')
        result_file = pd.read_csv(result_file_name)
        for j in range(len(result_keys)):
            vName = result_keys[j]
            mode = result_modes[vName]
            results_dataframe.at[i, vName] = get_result_value(result_file[vName], mode)
        model_name = parameter_json["model_name"]
        necess_params = necessary_parameters_per_model[model_name]
        results_dataframe.at[i,"# params"] =  get_number_params_per_model(parameter_json,necess_params,model_name)
    return results_dataframe

def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
#     print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
#     print(line_new)
#     print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
#         line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
#         print(line_new)
#     print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
#     print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("Trainable params:", num_trainable_params)
#     print("Non-trainable params:", total_params - num_trainable_params)
    return total_params
    
def get_best_value_idx(input_df,vName):
    best_result = {}
    best_index = input_df[vName].idxmax()
    best_value = max(input_df[vName])
    best_result[vName+'_idx'] = best_index
    best_result[vName] = best_value
    return best_result 

def get_number_params_per_model(parameter_json,necess_params,model_name):
#     with open(paramFile) as f:
#         parameter_json = json.load(f)
    config = parameter_json
    params_for_model = {}
    params_for_model["model_name"] = model_name
    for i in range(len(necess_params)):
        param_name = necess_params[i]
        params_for_model[param_name] = parameter_json[param_name]
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
    return gnn_model_summary(model)



training_properties = {}
training_properties


training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet2_opt/opt_007/ArmaNet_2layers_large1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet_2layers_large1"] = training_property


training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet3_opt/opt_001/ArmaNet3l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet3_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet4_opt/opt_001/ArmaNet4l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet4_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet5_opt/opt_001/ArmaNet5l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet5_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet6_opt/opt_001/ArmaNet5l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet6_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet7_opt/opt_001/ArmaNet7l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet7_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/ArmaNet8_opt/opt_001/ArmaNet8l_1"
training_property["parameter_keys"] = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]
training_properties["ArmaNet8_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet3_opt/opt_001/GCNNet_3l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet3_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet4_opt/opt_001/GCNNet_4l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet4_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet5_opt/opt_001/GCNNet_5l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet5_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet6_opt/opt_001/GCNNet_5l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet6_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet7_opt/opt_001/GCNNet_7l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet7_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet8_opt/opt_001/GCNNet_8l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet8_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/GCNNet9_opt/opt_001/GCNNet_9l"
training_property["parameter_keys"] = ["GCN::improved"]
training_properties["GCNNet9_opt001"] = training_property


# training_property = {}
# training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet1_opt/opt_001/TAGNet1l_1"
# training_property["parameter_keys"] = ["TAG::K_hops"]
# training_properties["TAGNet1_opt001"] = training_property

# training_property = {}
# training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet2_opt/opt_001/TAGNet4l_1"
# training_property["parameter_keys"] = ["TAG::K_hops"]
# training_properties["TAGNet2_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet3_opt/opt_001/TAGNet3l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet3_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet4_opt/opt_001/TAGNet4l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet4_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet5_opt/opt_001/TAGNet5l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet5_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet6_opt/opt_001/TAGNet6l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet6_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet7_opt/opt_001/TAGNet7l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet7_opt001"] = training_property

training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet8_opt/opt_001/TAGNet8l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet8_opt001"] = training_property


training_property = {}
training_property["path"] = "/home/nauck/joined_work/dataset_gnn/N020/ray_opt/TAGNet9_opt/opt_001/TAGNet9l_1"
training_property["parameter_keys"] = ["TAG::K_hops"]
training_properties["TAGNet9_opt001"] = training_property

training_directories_paths = {}
result_overview = {}


necessary_parameters_per_model = {}
necessary_parameters_per_model["ArmaNet_ray"] = ["final_linear_layer", "num_layers", "num_channels", "batch_norm_index", "ARMA::num_internal_layers", "ARMA::num_internal_stacks", "ARMA::dropout", "ARMA::shared_weights"]
necessary_parameters_per_model["TAGNet_ray"] = ["final_linear_layer", "num_layers", "num_channels", "batch_norm_index", "TAG::K_hops"]
necessary_parameters_per_model["GCNNet_ray"] = ["final_linear_layer", "num_layers", "num_channels", "batch_norm_index", "GCN::improved"]

for key in training_properties.keys():
    result = {}
    df_result = read_input_data(training_properties[key]["path"],30,training_properties[key]["parameter_keys"],necessary_parameters_per_model)
    result["df_result"] = df_result
    get_best_value_idx(df_result,"test_R2")
    best_result = get_best_value_idx(df_result,"test_R2")
    result["best_result"] = best_result
    result_overview[key] = result
    print(key,result_overview[key]["best_result"])



fig, ax = plt.subplots()
yVariable = "test_R2"
simDir = "ArmaNet3_opt001"
ax.scatter(np.array(result_overview[simDir]["df_result"]["# params"]), np.array(result_overview[simDir]["df_result"][yVariable]))
if yVariable == "test_R2":
    plt.ylim(.45, .85)
elif yVariable == "test_acc":
    plt.ylim(85,100)
plt.xlabel("number of parameters")
plt.ylabel(yVariable)
plt.title(simDir)