# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:12:00 2022

@author: tobia
"""
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from datasets.dataset import create_datasets, create_loaders
from training.training import run_training


def objective(cfg,trainloader,testloader,model,optimizer,engine):
    while True:
        losses, evaluations, final_eval = run_training(trainloader, testloader, engine, epochs=cfg["epochs"])
        session.report({"loss": final_eval[0]})  # Report to Tune