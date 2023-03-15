import logging
import torch
import json
#from torch_geometric.transforms import ToUndirected, Compose, RemoveIsolatedNodes, NormalizeScale
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, objective#, run_tuning
from datasets.dataset import create_datasets, create_loaders
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from utils.utils import  plot_loss, plot_R2, ImbalancedSampler, discrete_loss
import matplotlib.pyplot as plt
import numpy as np

#TO
import shutil
from ray import tune
from ray import air
import ray
import time
from sys import argv

#get time
start =time.time()

#Loading training configuration
with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)

if cfg['study::run'] == True:
    #arguments for ray
    temp_dir ='/p/tmp/tobiasoh/ray_tmp'
    N_gpus = 1
    N_cpus = int(argv[1])
    port_dashboard = int(argv[2])
    #init ray
    ray.init(_temp_dir=temp_dir,num_cpus=N_cpus, num_gpus = N_gpus, include_dashboard=True,dashboard_port=port_dashboard)
#save config in results
shutil.copyfile("configurations/configuration.json","results/configuration.json")

    
logging.basicConfig(filename=cfg['dataset::path'] + "results/regression.log", filemode="w", level=logging.INFO)

#Loading and pre-transforming data
#trainset, testset = create_datasets(cfg["dataset::path"], pre_transform=ToUndirected()) 
trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None)
trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg


#getting feature and target sizes
num_features = trainset.__getitem__(0).x.shape[1]
num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
num_targets = 1



#choosing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #TO device represents the 'device' on which a torch.tensor is placed (cpu or cuda) -> cuda uses gpus
#device = "cuda:0"
print(device)

#setting seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
numpy_seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#choosing criterion
criterion = torch.nn.MSELoss(reduction = 'mean')  #TO defines the loss
criterion.to(device)

#Runs study if set in configuration file
if cfg["study::run"]:
    #uses ray to run a study, to see functionality check training.objective
    
    search_space = {
        'layers' : tune.randint(cfg["study::layers_lower"],cfg["study::layers_upper"]),
        'HF' : tune.lograndint(cfg["study::hidden_features_lower"],cfg["study::hidden_features_upper"]),
        'heads' : tune.randint(cfg["study::heads_lower"],cfg["study::heads_upper"]),
        'LR' : tune.loguniform(cfg['study::lr::lower'],cfg['study::lr::upper']),
        #'batchsize' : tune.lograndint(cfg["study::batchsize_lower"],cfg["study::batchsize_upper"]),
        'dropout' : tune.quniform(cfg["study::dropout_lower"],cfg["study::dropout_upper"],0.01)
    }
    tune_config = tune.tune_config.TuneConfig(mode='min', metric='discrete_measure', num_samples = cfg['study::n_trials'])
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    tuner = tune.Tuner(tune.with_resources(tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features, 
                                            num_edge_features=num_edge_features, num_targets=num_targets, device=device, criterion=criterion),resources={"cpu": 2, "gpu":N_gpus/(N_cpus/2)}), param_space = search_space, 
tune_config=tune_config, 
run_config=run_config)
    results = tuner.fit()
    print(results)
    
    
    
else:
    params = {
        "num_layers" : cfg['num_layers'],
        "hidden_size" : cfg['hidden_size'],
        "dropout" : cfg["dropout"],
        "heads" : cfg['num_heads'],
        "num_features" : num_features,
        "num_edge_features" : num_edge_features,
        "num_targets" : num_targets
    }
    #Loading GNN model
    model = get_model(cfg, params)   #TO get_model does not load an old model but create a new one 
    model.to(device)

    #Choosing optimizer
    optimizer = get_optimizer(cfg, model)

    #Initializing engine
    engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"])


    losses, final_eval, output, labels = run_training(trainloader, testloader, engine, cfg)
    torch.save(list(output), "results/"  + f"output.pt") #saving train losses
    torch.save(list(labels), "results/"  + f"labels.pt") #saving train losses
    torch.save(list(losses), "results/"  + "losses1.pt") #saving train losses
    plt.plot(losses)

    logging.info("Final results:")
    logging.info(f"Accuracy: {final_eval[2]}")
    logging.info(f"R2: {final_eval[1]}")
    logging.info(f'Discrete loss: {final_eval[3]}')
    
    
    save_model = True
    if save_model:
        torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
        #torch.onnx.export(model,data,"supernode.onnx")

    torch.save(list(losses), "results/" + "losses.pt") #saving train losses
    plt.title(f'LR={cfg["optim::LR"]}')
    plt.plot(losses)
    """
    fig1,ax1=plt.subplots()
    x_ticks = np.array(range(2000))
    ax1.bar(x_ticks, labels[0])
    ax1.bar(x_ticks, output[0])
    ax1.set_title("Load Shed at Node")
    ax1.set_xlabel("Node ID")
    ax1.set_ylabel('Load Shed in p.U.')
    fig1.savefig("ac_node_feature_distr_active_power.png")
    """
end = time.time()
logging.info(f'\nOverall Runtime: {(end-start)/60} min')    



#torch.save(list(evaluations), "results/" + cfg["dataset::path"] + "evaluations.pt") #saving test evaluations

"""layers_lower=cfg["study::layers_lower"]
layers_upper=cfg["study::layers_upper"]
hidden_features_lower=cfg["study::hidden_features_lower"]
hidden_features_upper=cfg["study::hidden_features_upper"]
hidden_features_stride = cfg["study::hidden_features_stride"]
heads_lower = cfg["study::heads_lower"]
heads_upper = cfg["study::heads_upper"]
best_discrete_measure = np.Inf
for layers in range(layers_lower,layers_upper+1):
    for hidden_features in range(hidden_features_lower,hidden_features_upper+1, hidden_features_stride):
        for heads in range(heads_lower, heads_upper+1):
            logging.basicConfig(filename=f"results/regression_{layers}L_{hidden_features}HF.log", filemode="w", level=logging.INFO)
            logging.info('\n\n Starting new LR study with following Params:\n')
            logging.info(f'Layers: {layers}')
            logging.info(f'Hidden Features: {hidden_features}')
            logging.info(f'Heads: {heads}\n')
            #Setting model parameters (these are tunable in hyperoptimization)
            params = {
                "num_layers" : layers,
                "hidden_size" : hidden_features,
                "dropout" : cfg["dropout"],
                "heads" : heads,
                "num_features" : num_features,
                "num_edge_features" : num_edge_features,
                "num_targets" : num_targets
            }
            
            #Loading GNN model
            """
            
"""model = get_model(cfg, params)   #TO get_model does not load an old model but create a new one 
model.to(device)
           

#Choosing optimizer
optimizer = get_optimizer(cfg, model)

#Initializing engine
engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"])"""

"""
            objective = Objective(trainloader, testloader, cfg, params, device, criterion)
            optimal_params = run_tuning(cfg, objective)
        
            with open("optimal_params.json", "w+") as out:
                out.write(json.dumps(optimal_params))
        
            optim_lr=optimal_params["lr"]
            logging.info(f'\nResults of best LR ({optim_lr}) with\nLayers : {layers}\nHF : {hidden_features}\nHeads : {heads}\n')
            #TO added to start a new model after study and not keep training the model before
            model = get_model(cfg, params)   #TO get_model does not load an old model but create a new one 
            model.to(device)
            optimizer = get_optimizer(cfg, model)
            engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"])
            #End TO
            
            engine.optimizer.lr = optim_lr 
            losses, final_eval, output, labels = run_training(trainloader, testloader, engine, epochs=cfg["epochs"])
            #check if performance in discrete measure is better, if yes update best params
            if final_eval[3] < best_discrete_measure:
                best_eval = final_eval
                best_params = {'layers' : layers,
                               'hidden_features' : hidden_features,
                               'heads' : heads,
                               'lr' : optim_lr}
            torch.save(list(losses), "results/" + cfg["dataset::path"] + f"losses_{layers}L_{hidden_features}HF_{heads}heads_{optim_lr:.{3}f}lr.pt") #saving train losses
            torch.save(list(output), "results/" + cfg["dataset::path"] + f"output_{layers}L_{hidden_features}HF_{heads}heads_{optim_lr:.{3}f}lr.pt") #saving train losses
            torch.save(list(labels), "results/" + cfg["dataset::path"] + f"labels_{layers}L_{hidden_features}HF_{heads}heads_{optim_lr:.{3}f}lr.pt") #saving train losses
        
        
logging.info(f'Best overall parameters:\n{best_params}')
logging.info(f'Performance:')
logging.info(f'Final Loss {best_eval[0]}')
logging.info(f'Final R2 {best_eval[1]}')
logging.info(f'Final Accuracy {best_eval[2]}')
logging.info(f'Final Discrete Measure {best_eval[3]}')
"""
    
