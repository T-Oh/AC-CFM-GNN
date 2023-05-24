import logging
import torch
import json
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, objective  
from datasets.dataset import create_datasets, create_loaders, calc_mask_probs, save_node2vec
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
import matplotlib.pyplot as plt
import numpy as np

# TO
from utils.utils import weighted_loss_label, weighted_loss_var
import shutil
import ray
from ray import tune
from ray import air
from ray.tune.search.bayesopt import BayesOptSearch
import time
from sys import argv
from os.path import isfile
from torchmetrics import R2Score
from models.run_mean_baseline import run_mean_baseline
import os
from torch_geometric.nn import Node2Vec
from models.run_node2vec import run_node2vec



# get time
start = time.time()

# Loading training configuration
with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)
# choosing criterion
assert not (cfg['weighted_loss_label'] and cfg['weighted_loss_var']
            ), 'can not use both weighted losses at once'
assert not (cfg['crossvalidation'] and cfg['study::run']
            ), 'can only run a study or the crossvalidation not both'

# initialize ray
if cfg['study::run']:
    # arguments for ray
    TEMP_DIR = '/p/tmp/tobiasoh/ray_tmp'
    N_GPUS = 1
    N_CPUS = int(argv[1])
    port_dashboard = int(argv[2])
    # init ray
    ray.init(_temp_dir=TEMP_DIR, num_cpus=N_CPUS, num_gpus=N_GPUS,
             include_dashboard=True, dashboard_port=port_dashboard)

# save config in results
shutil.copyfile("configurations/configuration.json",
                "results/configuration.json")
logging.basicConfig(filename=cfg['dataset::path'] +
                    "results/regression.log", filemode="w", level=logging.INFO)

# Create Datasets and Dataloaders
print('Creating Datasets...')
trainset, testset, data_list = create_datasets(
    cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'])
t2 = time.time()
print(f'Creating datasets took {(start-t2)/60} mins')
print('Creating Dataloaders...')
trainloader, testloader = create_loaders(cfg, trainset, testset)
print(f'Creating dataloaders took {(t2-time.time())/60} mins')



# Calculate probabilities for masking of nodes if necessary
if cfg['use_masking'] or cfg['weighted_loss_var'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
    if isfile('node_label_vars.pt'):
        print('Using existing Node Label Variances for masking')
        mask_probs = torch.load('node_label_vars.pt')
    else:
        print('No node label variance file found\nCalculating Node Variances for Masking')
        mask_probs = calc_mask_probs(trainloader)
        torch.save(mask_probs, 'node_label_vars.pt')
else:
    #Masks are set to one in case it is wrongly used somewhere (when set to 1 masking results in multiplication with 1)
    mask_probs = torch.zeros(2000)+1


# getting feature and target sizes
num_features = trainset.__getitem__(0).x.shape[1]
num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
num_targets = 1


# choosing device
# TO device represents the 'device' on which a torch.tensor is placed (cpu or cuda) -> cuda uses gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cuda:0"
print(device)


# setting seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
numpy_seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Runs study if set in configuration file
if cfg["study::run"]:
    # uses ray to run a study, to see functionality check training.objective
    # set up search space
    search_space = {
        'layers': tune.quniform(cfg["study::layers_lower"], cfg["study::layers_upper"]+1, 1),
        'HF': tune.loguniform(cfg["study::hidden_features_lower"], cfg["study::hidden_features_upper"]+1),
        'heads': tune.quniform(cfg["study::heads_lower"], cfg["study::heads_upper"]+1, 1),
        'LR': tune.loguniform(cfg['study::lr::lower'], cfg['study::lr::upper']),
        'dropout': tune.quniform(cfg["study::dropout_lower"], cfg["study::dropout_upper"], 0.01),
        'gradclip': tune.quniform(cfg['study::gradclip_lower'], cfg['study::gradclip_upper'], 0.01),

        'reghead_size': tune.loguniform(cfg['study::reghead_size_lower'], cfg['study::reghead_size_upper']+1),
        'reghead_layers': tune.quniform(cfg["study::reghead_layers_lower"], cfg['study::reghead_layers_upper']+1, 1),
        # 'use_batchnorm'     : cfg['use_batchnorm'],
        'use_skipcon': float(cfg['use_skipcon']),
        'use_masking': float(cfg['use_masking']),
        'mask_bias': tune.quniform(cfg['study::mask_bias_lower'], cfg['study::mask_bias_upper']+0.1, 0.1),

        'loss_weight': tune.loguniform(cfg['study::loss_weight_lower'], cfg['study::loss_weight_upper'])
        # 'batchsize' : tune.lograndint(cfg["study::batchsize_lower"],cfg["study::batchsize_upper"])
    }
    # if cfg['study::batchnorm']:
    #    search_space['use_batchnorm'] = tune.choice([True, False])
    if cfg['study::skipcon']:
        search_space['use_skipcon'] = tune.uniform(
            0, 2)  # tune.choice([True, False])
    if cfg['study::masking']:
        search_space['use_masking'] = tune.uniform(
            0, 2)  # tune.choice([True, False])

    # set up optimizer and scheduler
    baysopt = BayesOptSearch(metric='r2', mode='max')
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration', metric='r2', mode='max', max_t=100, grace_period=10)
    # configurations
    tune_config = tune.tune_config.TuneConfig(
        num_samples=cfg['study::n_trials'], search_alg=baysopt, scheduler=scheduler)
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    # tuner
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features,
                             num_edge_features=num_edge_features, num_targets=num_targets, device=device, mask_probs=mask_probs),
        resources={"cpu": 1, "gpu": N_GPUS/(N_CPUS/1)}),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config)
    results = tuner.fit()
    print(results)


else:
    if cfg['model'] == 'Mean':  # Model used as baseline that simply predicts the mean load shed of the training set

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
        
        assert not cfg['train_set::shuffle'], 'Node2Vec can not be used with train_set::shuffle'
        print(f'Length training set:{len(trainset)}')
        #assert cfg['train_set::batchsize'] == len(trainset), 'Node2Vec can only be used with batchsize = length of the trainingset'
        embedding = run_node2vec(cfg, trainloader, device, data_list)

        exit()
        
            

    else:
        params = {
            "num_layers": cfg['num_layers'],
            "hidden_size": cfg['hidden_size'],
            "dropout": cfg["dropout"],
            "dropout_temp": cfg["dropout_temp"],
            "heads": cfg['num_heads'],
            "use_batchnorm": cfg['use_batchnorm'],
            "gradclip": cfg['gradclip'],
            "use_skipcon": cfg['use_skipcon'],
            "reghead_size": cfg['reghead_size'],
            "reghead_layers": cfg['reghead_layers'],
            "use_masking": cfg['use_masking'],
            "mask_probs": mask_probs,

            "num_features": num_features,
            "num_edge_features": num_edge_features,
            "num_targets": num_targets
        }

    # Init Criterion
    if cfg['weighted_loss_label']:
        criterion = weighted_loss_label(
            factor=torch.tensor(cfg['weighted_loss_factor']))
    elif cfg['weighted_loss_var']:
        criterion = weighted_loss_var(mask_probs, device)
    else:
        criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
    criterion.to(device)

    # Loadi GNN model
    model = get_model(cfg, params)
    model.to(device)
    
    # Init optimizer
    optimizer = get_optimizer(cfg, model)

    # Initializing engine
    engine = Engine(model, optimizer, device, criterion,
                    tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs)
    
    
    if cfg['crossvalidation']:
        folds = 7
        del trainset
        del testset
        trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = 1)
        trainloader, testloader = create_loaders(cfg, trainset, testset) 
        trainlosses = torch.zeros(folds)
        trainR2s = torch.zeros(folds)
        testlosses = torch.zeros(folds)
        testR2s = torch.zeros(folds)
    else:
        folds = 1
        
        
    for fold in range(folds):
        if fold > 0:
            del trainset
            del testset
            del trainloader
            del testloader
            del model
            del engine
            del output
            del labels
            os.rename('processed/', f'processed{int(fold)}')
            os.rename(f'processed{int(fold+1)}/', 'processed')
            trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1)
            trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg
            # ReInit GNN model
            model = get_model(cfg, params)
            model.to(device)
            
            # ReInit optimizer
            optimizer = get_optimizer(cfg, model)

            # ReInitializing engine
            engine = Engine(model, optimizer, device, criterion,
                            tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs)
        
        #Run Training
        metrics, final_eval, output, labels = run_training(
            trainloader, testloader, engine, cfg)
        
        #Save outputs, labels and losses of first fold
        if fold == 0:
            torch.save(list(output), "results/" + "output.pt")  # saving train losses
            torch.save(list(labels), "results/" + "labels.pt")  # saving train losses
            torch.save(list(metrics['train_loss']), "results/" + "train_losses.pt")  # saving train losses
            torch.save(list(metrics['test_loss']), "results/" + "test_losses.pt")  # saving train losses
            #Set variables for logging in case crossvalidation == False
            trainloss = torch.tensor(metrics['train_loss']).min()
            testloss = torch.tensor(metrics['test_loss']).min()
            trainR2 = torch.tensor(metrics['train_R2']).min()
            testR2 = torch.tensor(metrics['test_R2']).min()
            
            plt.plot(metrics['train_loss'])
            plt.plot(metrics['test_loss'])
        
        #Add results of fold to lists
        if cfg['crossvalidation']:
            trainlosses[fold] = torch.tensor(metrics['train_loss']).min()
            trainR2s[fold] = torch.tensor(metrics['train_R2']).max()
            testlosses[fold] = torch.tensor(metrics['test_loss']).min()
            testR2s[fold] = torch.tensor(metrics['test_R2']).max()
            trainloss = trainlosses.mean()
            trainR2 = trainR2s.mean()
            testloss = testlosses.mean()
            testR2 = testR2s.mean()
            result = {'trainloss' : trainlosses,
                      'trainR2' : trainR2s,
                      'testloss' : testlosses,
                      'testR2' : testR2s}
            
        #Save Metrics after last fold
        if fold == folds-1:
            logging.info("Final results:")
            logging.info(f'Train Loss: {trainloss}')
            logging.info(f"Train R2: {trainR2}")
            logging.info(f'Test Loss: {testloss}')
            logging.info(f'Test R2: {testR2}')

    save_model = True
    if save_model:
        torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
        # torch.onnx.export(model,data,"supernode.onnx")

end = time.time()
logging.info(f'\nOverall Runtime: {(end-start)/60} min')
