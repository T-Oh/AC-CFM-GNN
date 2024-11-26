# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:41:26 2023

@author: tobia
"""
import torch
import os
import logging

from os.path import isfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs, get_attribute_sizes
from models.get_models import get_model
from models.run_node2vec import run_node2vec
from utils.utils import setup_params, weighted_loss_label
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training


def run_crossval(cfg, device, N_CPUS=0):
    """
    runs the crossvalidation
    INPUT
        cfg     :   preloaded json configuration file
        device  :   torch.device
        
    """
    

    FOLDS = 7   #number of folds
    
    optim_metrics = {}
    optim_eval = {}
    trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = 1)
    if cfg['model'] == 'Node2Vec':
        trainloader, testloader = create_loaders(cfg, trainset, testset, Node2Vec=True)     #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
    else:
        if device == 'cuda' :   pin_memory=True
        else                :   pin_memory=False
        trainloader, testloader = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'])

    # Calculate probabilities for masking of nodes if necessary
    if cfg['use_masking']:
        if isfile('node_label_vars.pt'):
            print('Using existing Node Label Variances for masking')
            mask_probs = torch.load('node_label_vars.pt')
        else:
            print('No node label variance file found\nCalculating Node Variances for Masking')
            mask_probs = calc_mask_probs(trainloader, cfg)
            torch.save(mask_probs, 'node_label_vars.pt')
    else:
        #Masks are set to one in case it is wrongly used somewhere (when set to 1 masking results in multiplication with 1)
        mask_probs = torch.zeros(2000)+1

    # getting feature and target sizes
    num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)
    
    params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets)
    
    #Node2Vec
    if cfg['model'] == 'Node2Vec':
        
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
    

    # Init Criterion
    if cfg['task'] == 'GraphClass':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg['weighted_loss_label']:
        criterion = weighted_loss_label(
        factor=torch.tensor(cfg['weighted_loss_factor']))
    else:
        criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
    #criterion.to(device)

 
         
    for fold in range(FOLDS):
        print('\nSTARTING FOLD', fold)
        if fold > 0:
            del trainset, testset, trainloader, testloader, model, optimizer, engine, output, labels

            if not 'LDTSF' in cfg['data']:  #except for LDTSF the data has to be differently normalized for the different folds which are then saved in processed and processed 2-7
                os.rename('processed/', f'processed{int(fold)}')
                os.rename(f'processed{int(fold+1)}/', 'processed')

            if cfg['model'] == 'Node2Vec':
                trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1, embedding=normalized_embedding.to(device))
            else:
                trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1, data_type=cfg['data'], edge_attr=cfg['edge_attr'])
            trainloader, testloader = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'])                        #TO the loaders contain the data and get batchsize and shuffle from cfg
            # ReInit GNN model
        model = get_model(cfg, params)
        model.to(device)
        
        # ReInit optimizer
        optimizer = get_optimizer(cfg, model, params)
        
        #Init LR Scheduler
        LRScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001, verbose=True)
        
        # ReInitializing engine
        engine = Engine(model, optimizer, device, criterion,
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'])
        
        #Run Training
        metrics, final_eval, output, labels = run_training(
            trainloader, testloader, engine, cfg, LRScheduler, fold=fold)
        
        if fold == 0:
            for key in metrics.keys():
                if 'loss' in key:   
                    optim_metrics[key] = [np.min(metrics[key])]
                    optim_eval[key] = [np.min(final_eval[key])]
                elif np.array(metrics[key]).ndim > 1:               
                    optim_metrics[key] = [np.max(metrics[key], axis=0)]
                    optim_eval[key] = [np.max(final_eval[key], axis=0)]
                else:
                    optim_metrics[key] = [np.max(metrics[key])]
                    optim_eval[key] = [np.max(final_eval[key])]
        else:
            for key in metrics.keys():
                if 'loss' in key:   
                    optim_metrics[key].append(np.min(metrics[key]))
                    optim_eval[key].append(np.min(final_eval[key]))
                elif np.array(final_eval[key]).ndim > 1:
                    optim_metrics[key].append(np.max(metrics[key], axis=0))
                    optim_eval[key].append(np.max(final_eval[key], axis=0))
                else:               
                    optim_metrics[key].append(np.max(metrics[key]))
                    optim_eval[key].append(np.max(final_eval[key]))


        #Save outputs, labels and losses of first fold
        if fold == 0:
            torch.save(list(output), "results/" + "output_" + str(fold) + ".pt")  # saving train losses
            torch.save(list(labels), "results/" + "labels_" + str(fold) + ".pt")  # saving train losses

        
            
    #Save Metrics and model after last fold
    print(optim_eval)
    print(optim_metrics)
    for key in optim_metrics.keys():
        print(f"Mean optimal test {key}: {np.array(optim_eval[key]).mean()}")
        print(f"Mean optimal train {key}: {np.array(optim_metrics[key]).mean()}")

        logging.info("Final results:")
        logging.info(f"Mean optimal test {key}: {np.array(optim_eval[key]).mean()}")
        logging.info(f"Mean optimal train {key}: {np.array(optim_metrics[key]).mean()}")
        

    torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
        # torch.onnx.export(model,data,"supernode.onnx")"""
        