import logging
import torch
import json5
import shutil
import time
import os

import torch.multiprocessing

from numpy.random import seed as numpy_seed

from utils.utils import get_arg, check_config_conflicts
from utils.setups import setup_ProcessingConfig
from run.run_single import run_single
from run.run_crossval import run_crossval
from run.run_study import run_study
from normalization.normalize import normalize
from processing.process import run_processing
from datasets.dataset_graphlstm import create_lstm_datasets
from datasets.dataset import create_datasets

if __name__ == "__main__":
    #fix for windows
    torch.multiprocessing.freeze_support()

    start = time.time()
    print('NOT USING BUS TYPES AS FEATURES')    #ignoring bus type features in getitem of dataset

    # Loading training configuration
    configfile = "configurations/configuration.json"

    #Create results folder
    os.makedirs('results/plots', exist_ok=True)
    with open(configfile, "r") as io:
        cfg = json5.load(io)

    PATH = cfg["cfg_path"]

    #Pass Input Arguments
    N_TASKS = get_arg(1)
    N_CPUS_PER_TASK = get_arg(2)
    N_GPUS = get_arg(3)
    port_dashboard = get_arg(4)

    print('N_TASKS:', N_TASKS)
    print('N_CPUS_PER_TASK:', N_CPUS_PER_TASK)
    print('N_GPUS: ', N_GPUS, flush=True)

    #check for conflicts in configuration file
    check_config_conflicts(cfg)

    # save config in results for reference
    shutil.copyfile(PATH+"configurations/configuration.json", PATH+"results/configuration.json")
    logging.basicConfig(filename=PATH+ "results/regression.log", filemode="w", level=logging.INFO)

    # choosing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device, flush=True)

    # setting seeds
    torch.manual_seed(cfg["manual_seed"])
    torch.cuda.manual_seed(cfg["manual_seed"])
    numpy_seed(cfg["manual_seed"])
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    #Setup ProcessingConfig
    PROCESSING_CONFIG = setup_ProcessingConfig(cfg)

    #Process raw data
    if cfg['process']:
        run_processing(PROCESSING_CONFIG)

    #Create Datasets
    if cfg['data'] == 'LSTM':
        trainset, testset = create_lstm_datasets(cfg)
    else:
        trainset, testset, _ = create_datasets(cfg, normalized=False)   #use normalized False here as this only creates the sets for normalization

    #Normalize data if set in configuration file
    if cfg["normalize"]:
        print('Normalizing data...')
        normalize(PROCESSING_CONFIG, trainset, testset)

    # Runs study if set in configuration file
    if cfg["study::run"]:
        model = run_study(cfg, device, N_TASKS, N_CPUS_PER_TASK, N_GPUS, port_dashboard)

    #Runs crossvalidation
    elif cfg['crossvalidation']:
        model = run_crossval(cfg, device, N_CPUS=N_CPUS_PER_TASK)

    #Runs a single configuration
    else:
        model = run_single(cfg, device, N_CPUS=N_CPUS_PER_TASK)


    end = time.time()
    logging.info(f'\nOverall Runtime: {(end-start)/60} min')
