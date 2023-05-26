# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:41:26 2023

@author: tobia
"""
import torch
import os
import logging

from datasets.dataset import create_datasets, create_loaders
from models.get_models import get_model
from utils.utils import setup_params, weighted_loss_label, weighted_loss_var
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training


def run_crossval(cfg, device, mask_probs, num_features, num_edge_features):

    FOLDS = 7
    
    trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = 1)
    trainloader, testloader = create_loaders(cfg, trainset, testset) 
    trainlosses = torch.zeros(FOLDS)
    trainR2s = torch.zeros(FOLDS)
    testlosses = torch.zeros(FOLDS)
    testR2s = torch.zeros(FOLDS)
    params = setup_params(cfg, mask_probs, num_features, num_edge_features)
    
    # Init Criterion
    if cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor=torch.tensor(cfg['weighted_loss_factor']))
    elif cfg['weighted_loss_var']:
        criterion = weighted_loss_var(mask_probs, device)
    else:
        criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
    criterion.to(device)

     
     
    for fold in range(FOLDS):
        if fold > 0:
            del trainset, testset
            del trainloader
            del testloader
            del model
            del optimizer
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
        