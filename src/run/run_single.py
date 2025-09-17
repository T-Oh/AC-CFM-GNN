# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:29:21 2023

@author: tobia
"""

import torch


from torch.optim.lr_scheduler import ReduceLROnPlateau


from datasets.dataset import create_datasets, create_loaders, calc_mask_probs, get_attribute_sizes
from models.get_models import get_model
from models.run_mean_baseline import run_mean_baseline
from models.run_node2vec import run_node2vec
from utils.utils import setup_params, choose_criterion, save_params, setup_datasets_and_loaders, save_output, physics_loss
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training




def run_single(cfg, device, N_CPUS):
    """
    Trains a single model (i.e. no cross-validation and no study)
    """

    if cfg['model'] == 'Mean':  # Model used as baseline that simply predicts the mean load shed of the training set
        #Run Mean Baseline
        result = run_mean_baseline(cfg)
        exit()


    else:
        if device == 'cuda':    pin_memory = True
        else:                   pin_memory = False

        # Create Datasets and Dataloaders
        max_seq_len_LDTSF, trainset, trainloader, testloader = setup_datasets_and_loaders(cfg, N_CPUS, pin_memory)

        # Calculate probabilities for masking of nodes if necessary
        mask_probs = calc_mask_probs(trainloader, cfg)

        # getting feature sizes if datatype is not LDTSF
        num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)

        #Setup Parameter dictionary for Node2Vec (mask_probs, num_features and num_edge_features should be irrelevant)
        params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_seq_len_LDTSF)
        save_params(cfg['cfg_path'], params, 'single')

        #Node2Vec
        if cfg['model'] == 'Node2Vec':  trainloader, testloader, params = setup_node2vec(cfg, device, trainloader, mask_probs, params)

        criterion = choose_criterion(cfg['task'], cfg['weighted_loss_label'], cfg['weighted_loss_factor'], cfg, device)

        # Loading GNN model
        model = get_model(cfg, params)
        #model = model_  #torch.compile(model_)
        #model.to(device)

        # Init optimizer
        optimizer = get_optimizer(cfg, model, params)

        #Init LR Scheduler
        LRScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100, threshold=0.0001)

        # Initializing engine
        engine = Engine(model, optimizer, device, criterion,
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'], return_full_output=True)

        #Run Training
        _, _, output, labels, test_output, test_labels = run_training(trainloader, testloader, engine, cfg, LRScheduler)


        save_output(output, labels, test_output, test_labels)
        #Save Model
        torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")




def setup_node2vec(cfg, device, trainloader, mask_probs, params):
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
    num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]

            #Setup params for following task (MLP)
    params = setup_params(cfg, mask_probs, num_features, num_edge_features)
    return trainloader, testloader, params

        