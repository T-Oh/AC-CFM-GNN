# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:18:09 2023

@author: tobia
"""

import ray
import torch

from ray import tune, air
from os.path import isfile

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
from utils.utils import setup_searchspace, setup_params
from ray.tune.search.bayesopt import BayesOptSearch
from training.training import objective
from models.run_node2vec import run_node2vec



def run_study(cfg, device, N_CPUS, port_dashboard):
    print('\n\nRUN STUDY OUTPUT\n')
    # arguments for ray
    TEMP_DIR = '/p/tmp/tobiasoh/ray_tmp'
    N_GPUS = 1
    N_CPUS = N_CPUS
    port_dashboard = port_dashboard
    # init ray
    ray.init( _temp_dir=TEMP_DIR,num_cpus=N_CPUS, num_gpus=N_GPUS,
             include_dashboard=True, dashboard_port=port_dashboard)
    
    # Create Datasets and Dataloaders
    trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'])
    trainloader, testloader = create_loaders(cfg, trainset, testset)
   
    
    # getting feature and target sizes
    num_features = trainset.__getitem__(0).x.shape[1]
    num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]

    # Calculate probabilities for masking of nodes if necessary
    if cfg['use_masking'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
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
     
    #Node2Vec
    params = setup_params(cfg, mask_probs, num_features, num_edge_features)
    
    if cfg['model'] == 'Node2Vec':
        print('Creating Node2Vec Embedding')
        
        embedding = run_node2vec(cfg, trainloader, device, params, 0)
        normalized_embedding = embedding.data
        #Normalize the Embedding
        print(embedding.shape)
        for i in range(embedding.shape[1]):
            normalized_embedding[:,i] = embedding[:,i].data/embedding[:,i].data.max()
            
        # Create Datasets and Dataloaders
        trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], embedding=normalized_embedding.to(device))
        trainloader, testloader = create_loaders(cfg, trainset, testset)

       
        # getting feature and target sizes
        num_features = trainset.__getitem__(0).x.shape[1]
        print(f'New number of features: {num_features}')
        num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
        
        #Setup params for following task (MLP)
        params['num_features'] = num_features
    #Node2Vec End

    
    # uses ray to run a study, to see functionality check training.objective
    # set up search space
    search_space = setup_searchspace(cfg)


    # set up optimizer and scheduler
    baysopt = BayesOptSearch(metric='test_R2', mode='max')
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration', metric='test_R2', mode='max', max_t=100, grace_period=10)

    
    # configurations
    tune_config = tune.tune_config.TuneConfig(
        num_samples=cfg['study::n_trials'], search_alg=baysopt, scheduler=scheduler)
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    

    
    # tuner
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features,
                             num_edge_features=num_edge_features, num_targets=1, device=device, mask_probs=mask_probs),
        resources={"cpu": 1, "gpu": N_GPUS/(N_CPUS/1)}),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config)
    results = tuner.fit()
    print(results)