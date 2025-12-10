import logging
import torch
import json5
import shutil
import time
import os

from utils.utils import get_arg
from numpy.random import seed as numpy_seed

from run.run_single import run_single
from run.run_crossval import run_crossval
from run.run_study import run_study
from utils.utils import check_config_conflicts

if __name__ == "__main__":
    #fix for windows :\
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()


    # get time
    start = time.time()
    print('NOT USING BUS TYPES AS FEATURES')
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

    check_config_conflicts(cfg)

    # save config in results
    shutil.copyfile(PATH+"configurations/configuration.json", PATH+"results/configuration.json")
    logging.basicConfig(filename=PATH+ "results/regression.log", filemode="w", level=logging.INFO)


    # choosing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #device = "cuda:0"
    """device = torch.device('cpu')
    print('HARDCODED USING CPU!')
    print('HARDCODED USING CPU!')
    print('HARDCODED USING CPU!')"""
    print(device, flush=True)


    # setting seeds
    torch.manual_seed(cfg["manual_seed"])
    torch.cuda.manual_seed(cfg["manual_seed"])
    numpy_seed(cfg["manual_seed"])
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True



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
