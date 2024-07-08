import logging
import torch
import json
import shutil
import time

from sys import argv
from numpy.random import seed as numpy_seed

from run.run_single import run_single
from run.run_crossval import run_crossval
from run.run_study import run_study



# get time
start = time.time()

# Loading training configuration
configfile = "configurations/configuration.json"
with open(configfile, "r") as io:
    cfg = json.load(io)

#Pass Input Arguments
N_TASKS = int(argv[1])  #should only be >1 for studies -> controls the number of parallel trials
N_CPUS_PER_TASK = int(argv[2])   #controls the number of cpus per trial used as dataloaders (for run_single and run_crossval should be total number of cpus (used as loaders))
N_GPUS = int(argv[3])
port_dashboard = int(argv[4])
print('N_TASKS:', N_TASKS)
print('N_CPUS_PER_TASK:', N_CPUS_PER_TASK)  
print('N_GPUS: ', N_GPUS, flush=True)  

assert not (cfg['crossvalidation'] and cfg['study::run']), 'can only run a study or the crossvalidation not both'
assert not (cfg['data'] == 'DC' and cfg['stormsplit']>0), 'Stormsplit can only be used with AC data'
assert not (cfg['edge_attr'] == 'multi' and cfg['model'] == 'TAG'), 'TAG can only be used with Y as edge_attr not with multi'

# save config in results
shutil.copyfile("configurations/configuration.json", "results/configuration.json")
logging.basicConfig(filename=cfg['dataset::path'] + "results/regression.log", filemode="w", level=logging.INFO)




# choosing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #device = "cuda:0"
print(device)
 

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
