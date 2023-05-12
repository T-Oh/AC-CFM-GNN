import logging
import torch
import json
#from torch_geometric.transforms import ToUndirected, Compose, RemoveIsolatedNodes, NormalizeScale
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, objective#, run_tuning
from datasets.dataset import create_datasets, create_loaders, calc_mask_probs, mask_probs_add_bias
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
import matplotlib.pyplot as plt
import numpy as np

#TO
from utils.utils import weighted_loss_label, weighted_loss_var
import shutil
import ray
from ray import tune
from ray import air
from ray.tune.search.bayesopt import BayesOptSearch
import time
from sys import argv, exit
from os.path import isfile
from torchmetrics import R2Score
from models.run_mean_baseline import run_mean_baseline

#get time
start =time.time()

#Loading training configuration
with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)
#choosing criterion
assert not (cfg['weighted_loss_label'] and cfg['weighted_loss_var']), 'can not use both weighted losses at once'
assert not (cfg['crossvalidation'] and cfg['study::run']), 'can only run a study or the crossvalidation not both'

#initialize ray
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
trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'])
trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg


#Calculate probabilities for masking of nodes if necessary
if cfg['use_masking'] or cfg['weighted_loss_var'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
    if isfile('node_label_vars.pt'):
        print('Using existing Node Label Variances for masking')
        mask_probs = torch.load('node_label_vars.pt')
    else:
        print('No node label variance file found\nCalculating Node Variances for Masking')
        mask_probs= calc_mask_probs(trainloader)
        torch.save(mask_probs, 'node_label_vars.pt')
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


    

#Runs study if set in configuration file
if cfg["study::run"]:
    #uses ray to run a study, to see functionality check training.objective
    #set up search space
    search_space = {
        'layers'    : tune.quniform(cfg["study::layers_lower"],cfg["study::layers_upper"]+1,1),
        'HF'        : tune.loguniform(cfg["study::hidden_features_lower"],cfg["study::hidden_features_upper"]+1),
        'heads'     : tune.quniform(cfg["study::heads_lower"],cfg["study::heads_upper"]+1,1),
        'LR'        : tune.loguniform(cfg['study::lr::lower'],cfg['study::lr::upper']),
        'dropout'   : tune.quniform(cfg["study::dropout_lower"],cfg["study::dropout_upper"],0.01),
        'gradclip'  : tune.quniform(cfg['study::gradclip_lower'], cfg['study::gradclip_upper'],0.01),
        
        'reghead_size'      : tune.loguniform(cfg['study::reghead_size_lower'], cfg['study::reghead_size_upper']+1),
        'reghead_layers'    : tune.quniform(cfg["study::reghead_layers_lower"], cfg['study::reghead_layers_upper']+1,1),
        #'use_batchnorm'     : cfg['use_batchnorm'],
        'use_skipcon'       : float(cfg['use_skipcon']),
        'use_masking'       : float(cfg['use_masking']),
        'mask_bias'      : tune.quniform(cfg['study::mask_bias_lower'], cfg['study::mask_bias_upper']+0.1, 0.1),
            
        'loss_weight' : tune.loguniform(cfg['study::loss_weight_lower'], cfg['study::loss_weight_upper'])
        #'batchsize' : tune.lograndint(cfg["study::batchsize_lower"],cfg["study::batchsize_upper"])
    }
    #if cfg['study::batchnorm']:
    #    search_space['use_batchnorm'] = tune.choice([True, False])
    if cfg['study::skipcon']:
        search_space['use_skipcon'] = tune.uniform(0,2)#tune.choice([True, False])
    if cfg['study::masking']:
        search_space['use_masking'] = tune.uniform(0,2)#tune.choice([True, False])
        
    #set up optimizer and scheduler
    baysopt=BayesOptSearch(metric='r2', mode='max')
    scheduler = tune.schedulers.ASHAScheduler(time_attr='training_iteration', metric = 'r2', mode ='max', max_t=100, grace_period=10)
    #configurations
    tune_config = tune.tune_config.TuneConfig(num_samples = cfg['study::n_trials'], search_alg=baysopt, scheduler=scheduler)
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    #tuner
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features, 
                             num_edge_features=num_edge_features, num_targets=num_targets, device=device, mask_probs=mask_probs),
                            resources={"cpu": 1, "gpu":N_gpus/(N_cpus/1)}),
        param_space = search_space, 
        tune_config=tune_config, 
        run_config=run_config)
    results = tuner.fit()
    print(results)
    
    
    
else:
    if cfg['model'] == 'Mean':   #Model used as baseline that simply predicts the mean load shed of the training set
       
        result = run_mean_baseline(cfg)
        np.save('results/mean_result', result)
        if cfg['crossvalidation']:
            trainloss = result['trainloss'].mean()
            trainR2 = result['trainR2'].mean()
            testloss = result['testloss'].mean()
            testR2 = result['testR2'].mean()
            
        logging.info("Final results of Mean Baseline:")
        logging.info(f"Train Loss: {trainloss}")
        logging.info(f"Test Loss: {testloss}")
        logging.info(f"Train R2: {trainR2}")
        logging.info(f'Test R2: {testR2}')
        exit()
        
    elif cfg['model'] == 'Node2Vec':

        params={
            "edge_index"      : next(iter(trainloader)).edge_index,
            "embedding_dim"   : 32,
            "walk_length"     : 10,
            "context_size"    : 1,
            "walks_per_node"  : 1
            }
        
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

    #Choose Criterion
    if cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor = torch.tensor(cfg['weighted_loss_factor']))
    elif cfg['weighted_loss_var']:
        criterion = weighted_loss_var(mask_probs, device)    
    else:    
        criterion = torch.nn.MSELoss(reduction = 'mean')  #TO defines the loss
    criterion.to(device)

    #Loading GNN model
    model = get_model(cfg, params)   #TO get_model does not load an old model but create a new one 
    model.to(device)
    #Choosing optimizer
    optimizer = get_optimizer(cfg, model)
    
    #Initializing engine
    engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"], var=mask_probs)

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

    
