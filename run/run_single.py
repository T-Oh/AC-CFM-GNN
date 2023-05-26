# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:29:21 2023

@author: tobia
"""
import numpy as np
import torch
import logging
from models.get_models import get_model
from models.run_mean_baseline import run_mean_baseline
from models.run_node2vec import run_node2vec
from utils.utils import weighted_loss_label, weighted_loss_var, setup_params
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training

def run_single(cfg, trainloader, testloader, device, mask_probs, num_features, num_edge_features):
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


    else:
        #Setup Parameter dictionary
        params = setup_params(cfg, mask_probs, num_features, num_edge_features)
        
        if cfg['model'] == 'Node2Vec':
            
            embedding = run_node2vec(cfg, trainloader, device, params, 0)
            #Normalize the Embedding
            print(embedding.shape)
            for i in range(embedding.shape[1]):
                embedding[:,i] = embedding[:,i]/embedding[:,i].max
                          
            """save_model = True
            if save_model:
                torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")"""
  
        else:
    
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
            
            #Run Training
            metrics, final_eval, output, labels = run_training(
                trainloader, testloader, engine, cfg)
            
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
