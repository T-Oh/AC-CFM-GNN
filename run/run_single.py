# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:29:21 2023

@author: tobia
"""
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from os.path import isfile
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
from models.get_models import get_model
from models.run_mean_baseline import run_mean_baseline
from models.run_node2vec import run_node2vec
from utils.utils import weighted_loss_label, setup_params
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training

def run_single(cfg, device):
    """
    Trains a single model (i.e. no cross-validation and no study)
    """

    if cfg['model'] == 'Mean':  # Model used as baseline that simply predicts the mean load shed of the training set
             
        #Run Mean Baseline
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


    else:
        
        # Create Datasets and Dataloaders
        trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'])
        if cfg['model'] == 'Node2Vec':
             trainloader, testloader = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
        else:
             trainloader, testloader = create_loaders(cfg, trainset, testset)

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
            mask_probs = torch.zeros(trainset.__getitem__(0).x.shape[0])+1


        # getting feature and target sizes
        num_features = trainset.__getitem__(0).x.shape[1]
        num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
        
        #Setup Parameter dictionary for Node2Vec (mask_probs, num_features and num_edge_features should be irrelevant)
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
                          

        #Regular Models (GINE, GAT, TAG, MLP)            
        # Init Criterion
        if cfg['weighted_loss_label']:
            criterion = weighted_loss_label(
                factor=torch.tensor(cfg['weighted_loss_factor']))
        else:
            criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
        criterion.to(device)
    
        # Loadi GNN model
        model = get_model(cfg, params)
        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('NUMBER OF PARAMETERS:')
        print(pytorch_total_params)
        
        # Init optimizer
        optimizer = get_optimizer(cfg, model, params)
        
        #Init LR Scheduler
        LRScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100, threshold=0.0001, verbose=True)
    
        # Initializing engine
        engine = Engine(model, optimizer, device, criterion,
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'])
        
        #Run Training
        metrics, final_eval, output, labels = run_training(
            trainloader, testloader, engine, cfg, LRScheduler)
        
        #Save outputs, labels and losses of first fold
        torch.save(list(output), "results/" + "output.pt")  # saving train losses
        torch.save(list(labels), "results/" + "labels.pt")  # saving train losses
        torch.save(list(metrics['train_loss']), "results/" + "train_losses.pt")  # saving train losses
        torch.save(list(metrics['test_loss']), "results/" + "test_losses.pt")  # saving train losses
        #Set variables for logging in case crossvalidation == False
        trainloss = torch.tensor(metrics['train_loss']).min()
        testloss = torch.tensor(metrics['test_loss']).min()
        trainR2 = torch.tensor(metrics['train_R2']).min()
        testR2 = torch.tensor(metrics['test_R2']).min()
            
            
        torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
        
        if device.type != 'cuda':
            plt.plot(metrics['train_loss'])
            plt.plot(metrics['test_loss'])
