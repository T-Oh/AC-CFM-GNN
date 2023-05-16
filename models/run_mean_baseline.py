# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:02:30 2023

@author: tobia
"""
from datasets.dataset import create_datasets, create_loaders
import os

import torch
from torchmetrics import R2Score

def run_mean_baseline(cfg):
    """
    Runs the Meanbaseline (mean of every node)
    Calculates the mean of every node in the train set and uses this as predictions

    Parameters
    ----------
    cfg : config file

    Returns
    -------
    result : dict of lists trainloss, testloss, trainR2, testR2 
            depending on wether crossvalidation is used or not returns floats or list of floats for all folds

    """

    if cfg['crossvalidation']:
        folds = 7
        trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = 1)
        trainloader, testloader = create_loaders(cfg, trainset, testset) 
        trainlosses = torch.zeros(folds)
        trainR2s = torch.zeros(folds)
        testlosses = torch.zeros(folds)
        testR2s = torch.zeros(folds)
    else:
        folds = 1
        
        
    for fold in range(folds):
        print(fold)
        if fold > 0:
            os.rename('processed/', f'processed{int(fold)}')
            os.rename(f'processed{int(fold+1)}/', 'processed')
            trainset, testset, _ = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None, stormsplit = fold+1)
            trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg
            
        #calculate means to pass to model
        means = torch.zeros(2000)
        train_labels = torch.zeros(len(trainset), 2000)
        train_output = torch.zeros(len(trainset), 2000)
        test_labels = torch.zeros(len(testset), 2000)

        #Init metrics
        R2 = R2Score()
        criterion = torch.nn.MSELoss(reduction='mean')
             
        #compile labels and calculate means
        index = 0
        for i, batch in enumerate(trainloader):
            N_instances = int(len(batch.node_labels)/2000)
            train_labels[index:index+N_instances] = batch.node_labels.reshape(N_instances,2000)
            index = index+N_instances 
        means = train_labels.mean(dim=0)
    
        #Compile output of Trainset
        for i in range(len(trainset)):
            train_output[i] = means
            
        #calc loss and R2
        trainloss = criterion(train_output.reshape(-1), train_labels.reshape(-1))
        trainR2 = R2(train_output.reshape(-1), train_labels.reshape(-1))
        
        #Compile labels and output of Testset
        index = 0
        for i, batch in enumerate(testloader):
            N_instances = int(len(batch.node_labels)/2000)
            test_labels[index:index+N_instances] = batch.node_labels.reshape(N_instances,2000)
            index = index+N_instances
        test_output = train_output[:len(testset)]
        
        #save the means and some labels
        torch.save(list(means), "results/"  + "means.pt") #saving train losses
        torch.save(list(train_labels[0:16000]), "results/"  + "labels.pt") #saving train losses
    
        #calc loss and R2
        trainloss = criterion(train_output.reshape(-1), train_labels.reshape(-1))
        trainR2 = R2(train_output.reshape(-1), train_labels.reshape(-1))
        testloss = criterion(test_output.reshape(-1), test_labels.reshape(-1))
        testR2 = R2(test_output.reshape(-1), test_labels.reshape(-1))
        
        
        #In case of crossvalidation save to tensors
        if folds > 1:
            trainlosses[fold] = trainloss
            trainR2s[fold] = trainR2
            testlosses[fold] = testloss
            testR2s[fold] = testR2
            trainloss = trainlosses.mean()
            trainR2 = trainR2s.mean()
            testloss = testlosses.mean()
            testR2 = testR2s.mean()
            result = {'trainloss' : trainlosses,
                      'trainR2' : trainR2s,
                      'testloss' : testlosses,
                      'testR2' : testR2s}
        else: 
            result = {'trainloss' : trainloss,
                      'trainR2' : trainR2,
                      'testloss' : testloss,
                      'testR2' : testR2}
            
    return result