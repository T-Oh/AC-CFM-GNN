# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:03:35 2022

@author: tobia
"""

import logging
import torch
import json
#from torch_geometric.transforms import ToUndirected, Compose, RemoveIsolatedNodes, NormalizeScale
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, run_tuning, Objective
from datasets.dataset import create_datasets, create_loaders
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from utils.utils import  plot_loss, plot_R2, ImbalancedSampler

#TO
#import hiddenlayer as hl #for GNN visualization
import shutil
#from ray import tune, air
#from ray.air import session
#from ray.tune.search.optuna import OptunaSearch


#save config in results
shutil.copyfile("configurations/configuration.json","results/configuration.json")



logging.basicConfig(filename="results/regression.log", filemode="w", level=logging.INFO)

#Loading training configuration
with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)


#Loading and pre-transforming data
trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None)
print(trainset)