import logging
import torch
import json
#from torch_geometric.transforms import ToUndirected, Compose, RemoveIsolatedNodes, NormalizeScale
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, objective#, run_tuning
from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
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
    ray.init(_temp_dir=temp_dir, num_cpus=N_cpus, num_gpus = N_gpus, include_dashboard=True, dashboard_port=port_dashboard) 
#save config in results
shutil.copyfile("configurations/configuration.json","results/configuration.json")

    
logging.basicConfig(filename=cfg['dataset::path'] + "results/regression.log", filemode="w", level=logging.INFO)

#Loading and pre-transforming data
#trainset, testset = create_datasets(cfg["dataset::path"], pre_transform=ToUndirected()) 
trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None)
trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg

#Calculate probabilities for masking of nodes if necessary
if cfg['use_masking']:
    mask_probs = calc_mask_probs(trainloader, cfg['masking_bias'])
else:
    mask_probs = torch.ones(2000)

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
        'layers'    : 4,#tune.qrandint(cfg["study::layers_lower"],cfg["study::layers_upper"]+1,1),
        'HF'        : tune.lograndint(cfg["study::hidden_features_lower"],cfg["study::hidden_features_upper"]+1),
        'heads'     : tune.qrandint(cfg["study::heads_lower"],cfg["study::heads_upper"]+1,1),
        'LR'        : tune.loguniform(cfg['study::lr::lower'],cfg['study::lr::upper']),
        'dropout'   : 0.0,#tune.quniform(cfg["study::dropout_lower"],cfg["study::dropout_upper"],0.01),
        'gradclip'  : tune.quniform(cfg['study::gradclip_lower'], cfg['study::gradclip_upper'],0.01),
        'dropout_off_epoch' : 1000,# tune.quniform(cfg['study::dropout_off_epoch_lower'], cfg['study::dropout_off_epoch_lower'], 100),
        'reghead_size'      : tune.lograndint(cfg['study::reghead_size_lower'], cfg['study::reghead_size_upper']+1),
        'reghead_layers'    : tune.qrandint(cfg["study::reghead_layers_lower"], cfg['study::reghead_layers_upper']+1,1),
        'use_batchnorm'     : cfg['use_batchnorm'],
        'use_skipcon'       : cfg['use_skipcon']
        #'batchsize' : tune.lograndint(cfg["study::batchsize_lower"],cfg["study::batchsize_upper"])
    }
    if cfg['study::batchnorm']:
        search_space['use_batchnorm'] = tune.choice([True, False])
    if cfg['study::skipcon']:
        search_space['use_skipcon'] = tune.choice([True, False])
    tune_config = tune.tune_config.TuneConfig(mode='max', metric='r2', num_samples = cfg['study::n_trials'])
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    tuner = tune.Tuner(tune.with_resources(tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features, 
                                            num_edge_features=num_edge_features, num_targets=num_targets, device=device, criterion=criterion),resources={"cpu": 1, "gpu":N_gpus/(N_cpus/1)}), param_space = search_space, 
tune_config=tune_config, 
run_config=run_config)
    results = tuner.fit()
    print(results)
    
    
    
else:
    params = {
        "num_layers"    : cfg['num_layers'],
        "hidden_size"   : cfg['hidden_size'],
        "dropout"       : cfg["dropout"],
        "dropout_temp"  : cfg["dropout_temp"],
        "heads"         : cfg['num_heads'],
        "use_batchnorm" : cfg['use_batchnorm'],
        "gradclip"      : cfg['gradclip'],
        "use_skipcon"   : cfg['use_skipcon'],
        "reghead_size"  : cfg['reghead_size'],
        "reghead_layers": cfg['reghead_layers'],
        "use_masking"   : cfg['use_masking'],
        "mask_probs"    : mask_probs,
        
        "num_features"  : num_features,
        "num_edge_features" : num_edge_features,
        "num_targets"   : num_targets
    }
    #Loading GNN model
    model = get_model(cfg, params)   #TO get_model does not load an old model but create a new one 
    model.to(device)

    #Choosing optimizer
    optimizer = get_optimizer(cfg, model)

    #Initializing engine
    engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"], mask_probs=mask_probs)

    losses, final_eval, output, labels = run_training(trainloader, testloader, engine, cfg)
    torch.save(list(output), "results/"  + f"output.pt") #saving train losses
    torch.save(list(labels), "results/"  + f"labels.pt") #saving train losses
    torch.save(list(losses), "results/"  + "losses1.pt") #saving train losses
    plt.plot(losses)

    logging.info("Final results:")
    logging.info(f"Accuracy: {final_eval[2]}")
    logging.info(f"R2: {final_eval[1]}")
    logging.info(f'Discrete loss: {final_eval[3]}')
    print(f'Final R2: {final_eval[1]}')
    
    
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

    
