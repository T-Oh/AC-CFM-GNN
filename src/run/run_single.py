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
from utils.utils import choose_criterion, save_params, save_output
from utils.setups import setup_datasets_and_loaders, setup_params
from utils.get_optimizers import get_optimizer
from training.engine import Engine
from training.training import run_training

def run_single(cfg, device, N_CPUS):
    """
    Run a single model training and evaluation pipeline.
    This function orchestrates the complete workflow for training and evaluating a GNN model,
    including dataset setup, model initialization, training execution, and result saving.
    Parameters
    ----------
    cfg : dict
            Configuration dictionary containing model parameters, task settings, data paths,
            and hyperparameters for training.
    cfg['model'] : str
            The model type to use. Supported values include 'Mean' (baseline) and 'Node2Vec'.
    cfg['task'] : str
            The type of task to perform (e.g., classification, regression).
    cfg['use_masking'] : bool
            Whether to apply node masking during training.
    cfg['mask_bias'] : float
            Bias parameter for masking.
    cfg['weighted_loss_label'] : str
            Label for weighted loss calculation.
    cfg['weighted_loss_factor'] : float
            Factor for weighting the loss.
    cfg['track_gradients'] : bool
            Whether to track gradients during training.
    cfg['track_test_gradients'] : bool
            Whether to track gradients during testing.
    cfg['accuracy_tolerance'] : float
            Tolerance threshold for early stopping.
    cfg['cfg_path'] : str
            Path to save configuration and parameters.
    device : str
            Computing device to use ('cuda' or 'cpu').
    N_CPUS : int
            Number of CPUs to use for data loading.
    Returns
    -------
    None
            Results are saved to disk via save_output() and model weights are saved to "results/".
    Notes
    -----
    For 'Mean' baseline model, the function exits after execution.
    For other models, the function sets up datasets, trains the model, and saves outputs.
    """

    if cfg['model'] == 'Mean':  # Model used as baseline that simply predicts the mean load shed of the training set
        #Run Mean Baseline
        run_mean_baseline(cfg)
        exit()


    else:
        if device == 'cuda':    pin_memory = True
        else:                   pin_memory = False

        # Create Datasets and Dataloaders
        max_seq_len_LDTSF, trainset, trainloader, testloader, PROCESSING_LSTM_DATA = setup_datasets_and_loaders(cfg, N_CPUS, pin_memory)
        if PROCESSING_LSTM_DATA:
                print('Processing Successful \n After normalization using normalize_GTSF.py or normalize_GTSF_PU.py, switch to one of the LSTM models and run the main again.')
                return
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

        # Init optimizer
        optimizer = get_optimizer(cfg, model, params)

        #Init LR Scheduler
        LRScheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, threshold=0.001)


        # Initializing engine
        engine = Engine(model, optimizer, device, criterion,
                        tol=cfg["accuracy_tolerance"], task=cfg["task"], var=mask_probs, masking=cfg['use_masking'], mask_bias=cfg['mask_bias'],
                         return_full_output=True, track_gradients=cfg['track_gradients'], track_test_gradients=cfg['track_test_gradients'])

        #Run Training
        _, _, output, labels, test_output, test_labels = run_training(trainloader, testloader, engine, cfg, LRScheduler)


        save_output(output, labels, test_output, test_labels)
        #Save Model
        torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")




def setup_node2vec(cfg, device, trainloader, mask_probs, params):
    """Set up Node2Vec embeddings and integrate them into the dataset for training.
    This function generates Node2Vec embeddings from the training data, normalizes them,
    and creates new datasets and dataloaders that incorporate these embeddings as node features
    for downstream model training.
    Configuration dictionary containing:
    cfg['dataset::path'] : str
            Path to the dataset.
    cfg['stormsplit'] : bool
            Whether to apply storm-based data splitting.
    trainloader : DataLoader
            PyTorch DataLoader for training data containing graph objects.
    mask_probs : torch.Tensor or None
            Masking probabilities for nodes, passed to setup_params.
    params : dict
            Parameter dictionary to be updated with new feature sizes.
    tuple of (DataLoader, DataLoader, dict)
            trainloader : DataLoader
                    Updated DataLoader for training data with Node2Vec embeddings integrated.
            testloader : DataLoader
                    DataLoader for test data with Node2Vec embeddings integrated.
            params : dict
                    Updated parameter dictionary containing new num_features and num_edge_features
                    based on the dataset with integrated embeddings.
    The Node2Vec embeddings are normalized per feature dimension by dividing by the maximum
    value in that dimension. The embeddings are moved to the specified device before integration
    into the dataset."""
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

        