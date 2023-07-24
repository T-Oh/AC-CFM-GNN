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

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
from models.get_models import get_model
from models.run_node2vec import run_node2vec
from utils.utils import setup_params, weighted_loss_label
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training


def run_crossval(cfg, device):

    FOLDS = 7
    
    trainlosses = torch.zeros(FOLDS)
    trainR2s = torch.zeros(FOLDS)
    testlosses = torch.zeros(FOLDS)
    testR2s = torch.zeros(FOLDS)
    trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = 1)
    if cfg['model'] == 'Node2Vec':
        trainloader, testloader = create_loaders(cfg, trainset, testset, Node2Vec=True) 
    else:
        trainloader, testloader = create_loaders(cfg, trainset, testset)

    # Calculate probabilities for masking of nodes if necessary
    if cfg['use_masking']:
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
    
    params = setup_params(cfg, mask_probs, num_features, num_edge_features)
    
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
    if cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor=torch.tensor(cfg['weighted_loss_factor']))
    else:
        criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
    criterion.to(device)

 
         
    for fold in range(FOLDS):
        if fold > 0:
            del trainset, testset, trainloader, testloader, model, optimizer, engine, output, labels
            os.rename('processed/', f'processed{int(fold)}')
            os.rename(f'processed{int(fold+1)}/', 'processed')
            if cfg['model'] == 'Node2Vec':
                trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1, embedding=normalized_embedding.to(device))
            else:
                trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1)
            trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg
            # ReInit GNN model
        model = get_model(cfg, params)
        model.to(device)
        
        # ReInit optimizer
        optimizer = get_optimizer(cfg, model)
        
        #Init LR Scheduler
        LRScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001, verbose=True)
        
        # ReInitializing engine
        engine = Engine(model, optimizer, device, criterion,
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'])
        
        #Run Training
        metrics, final_eval, output, labels = run_training(
            trainloader, testloader, engine, cfg, LRScheduler)
        
        #Save outputs, labels and losses of first fold
        if fold == 0:
            torch.save(list(output), "results/" + "output.pt")  # saving train losses
            torch.save(list(labels), "results/" + "labels.pt")  # saving train losses
            torch.save(list(metrics['train_loss']), "results/" + "train_losses.pt")  # saving train losses
            torch.save(list(metrics['test_loss']), "results/" + "test_losses.pt")  # saving train losses
            #Set variables for logging in case crossvalidation == False
    
        
        #Add results of fold to lists
        trainlosses[fold] = torch.tensor(metrics['train_loss']).min()
        trainR2s[fold] = torch.tensor(metrics['train_R2']).max()
        testlosses[fold] = torch.tensor(metrics['test_loss']).min()
        testR2s[fold] = torch.tensor(metrics['test_R2']).max()
        result = {'trainloss' : trainlosses,
                  'trainR2' : trainR2s,
                  'testloss' : testlosses,
                  'testR2' : testR2s}
        torch.save(result, 'results/crossval_results.pt')
        
            
    #Save Metrics and model after last fold

    trainloss = trainlosses.mean()
    trainR2 = trainR2s.mean()
    testloss = testlosses.mean()
    testR2 = testR2s.mean()
    logging.info("Final results:")
    logging.info(f'Train Loss: {trainloss}')
    logging.info(f"Train R2: {trainR2}")
    logging.info(f'Test Loss: {testloss}')
    logging.info(f'Test R2: {testR2}')
    torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
        # torch.onnx.export(model,data,"supernode.onnx")
        