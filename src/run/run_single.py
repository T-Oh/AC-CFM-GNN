# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:29:21 2023

@author: tobia
"""

import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs, get_attribute_sizes
from datasets.dataset_graphlstm import create_lstm_datasets, create_lstm_dataloader
from models.get_models import get_model
from models.run_mean_baseline import run_mean_baseline
from models.run_node2vec import run_node2vec
from utils.utils import weighted_loss_label, setup_params
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
        max_seq_length = -1
        if cfg['model'] == 'Node2Vec':
             trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
             trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
        elif cfg['model'] == 'GATLSTM':
            # Split dataset into train and test indices
            trainset, testset = create_lstm_datasets(cfg["dataset::path"], cfg['train_size'], cfg['manual_seed'])
            # Create DataLoaders for train and test sets
            trainloader = create_lstm_dataloader(trainset, batch_size=cfg['train_set::batchsize'], shuffle=True)
            testloader = create_lstm_dataloader(testset, batch_size=cfg['test_set::batchsize'], shuffle=False)
        else:
             trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
             trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'], task=cfg['task'])

        # Calculate probabilities for masking of nodes if necessary
        mask_probs = calc_mask_probs(trainloader, cfg)


        # getting feature sizes if datatype is not LDTSF
        num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)

        #Setup Parameter dictionary for Node2Vec (mask_probs, num_features and num_edge_features should be irrelevant)
        params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_seq_length)

        #Node2Vec
        if cfg['model'] == 'Node2Vec':  trainloader, testloader, params = setup_node2vec(cfg, device, trainloader, mask_probs, params)


        #Regular Models (GINE, GAT, TAG, MLP, GraphTransformer)
        # Init Criterion
        if cfg['task'] in ['GraphClass', 'typeIIClass']:
            criterion = torch.nn.CrossEntropyLoss()
        elif cfg['weighted_loss_label']:
            criterion = weighted_loss_label(
            factor=torch.tensor(cfg['weighted_loss_factor']))
        else:
            criterion = torch.nn.MSELoss(reduction='mean')  # TO defines the loss
        #criterion.to(device)

        # Loading GNN model
        model_ = get_model(cfg, params)
        model = model_  #torch.compile(model_)
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
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'], return_full_output=True)

        #Run Training
        _, _, output, labels, test_output, test_labels = run_training(trainloader, testloader, engine, cfg, LRScheduler)

        #Save outputs, labels and losses of first fold
        torch.save(output, "results/" + "output.pt")  # saving train losses
        torch.save(labels, "results/" + "labels.pt")  # saving train losses
        torch.save(output, "results/" + "test_output.pt")  # saving train losses
        torch.save(labels, "results/" + "test_labels.pt")  # saving train losses
        #torch.save(list(metrics['train_loss']), "results/" + "train_losses.pt")  # saving train losses
        #torch.save(list(metrics['test_loss']), "results/" + "test_losses.pt")  # saving train losses
    
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

        