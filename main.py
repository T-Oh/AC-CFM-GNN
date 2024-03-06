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
N_CPUS = int(argv[1])
port_dashboard = int(argv[2])
    

assert not (cfg['crossvalidation'] and cfg['study::run']), 'can only run a study or the crossvalidation not both'
assert not (cfg['data'] == 'DC' and cfg['stormsplit']>0), 'Stormsplit can only be used with AC data'


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
    model = run_study(cfg, device, N_CPUS, port_dashboard)
    #model = run_study(cfg, device, 1, 8123)

#Runs crossvalidation
elif cfg['crossvalidation']:
    model = run_crossval(cfg, device, N_CPUS=N_CPUS)

#Runs a single configuration    
else:
    model = run_single(cfg, device, N_CPUS=N_CPUS)



end = time.time()
logging.info(f'\nOverall Runtime: {(end-start)/60} min')
