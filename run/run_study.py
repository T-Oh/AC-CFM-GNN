# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:18:09 2023

@author: tobia
"""

import ray

from sys import argv
from ray import tune, air

from utils.utils import setup_searchspace
from ray.tune.search.bayesopt import BayesOptSearch
from training.training import objective


def run_study(cfg, trainloader, testloader, device, mask_probs, num_features, num_edge_features):

    # arguments for ray
    TEMP_DIR = '/p/tmp/tobiasoh/ray_tmp'
    N_GPUS = 1
    N_CPUS = int(argv[1])
    port_dashboard = int(argv[2])
    # init ray
    ray.init(_temp_dir=TEMP_DIR, num_cpus=N_CPUS, num_gpus=N_GPUS,
             include_dashboard=True, dashboard_port=port_dashboard)
    
    # uses ray to run a study, to see functionality check training.objective
    # set up search space
    search_space = setup_searchspace(cfg)

    # set up optimizer and scheduler
    baysopt = BayesOptSearch(metric='r2', mode='max')
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration', metric='r2', mode='max', max_t=100, grace_period=10)
    # configurations
    tune_config = tune.tune_config.TuneConfig(
        num_samples=cfg['study::n_trials'], search_alg=baysopt, scheduler=scheduler)
    run_config = air.RunConfig(local_dir=cfg['dataset::path']+'results/')
    # tuner
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(objective, trainloader=trainloader, testloader=testloader, cfg=cfg, num_features=num_features,
                             num_edge_features=num_edge_features, num_targets=1, device=device, mask_probs=mask_probs),
        resources={"cpu": 1, "gpu": N_GPUS/(N_CPUS/1)}),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config)
    results = tuner.fit()
    print(results)