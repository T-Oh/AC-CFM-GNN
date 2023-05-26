import logging
import torch
import json
import shutil
import time

from numpy.random import seed as numpy_seed
from os.path import isfile

from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
from run.run_single import run_single
from run.run_crossval import run_crossval
from run.run_study import run_study



# get time
start = time.time()

# Loading training configuration
configfile = "configurations/configuration.json"
with open(configfile, "r") as io:
    cfg = json.load(io)
    
# choosing criterion
assert not (cfg['weighted_loss_label'] and cfg['weighted_loss_var']), 'can not use both weighted losses at once'
assert not (cfg['crossvalidation'] and cfg['study::run']), 'can only run a study or the crossvalidation not both'


# save config in results
shutil.copyfile("configurations/configuration.json", "results/configuration.json")
logging.basicConfig(filename=cfg['dataset::path'] + "results/regression.log", filemode="w", level=logging.INFO)

# Create Datasets and Dataloaders
print('Creating Datasets...')
trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'])
t2 = time.time()
print(f'Creating datasets took {(start-t2)/60} mins')
print('Creating Dataloaders...')
trainloader, testloader = create_loaders(cfg, trainset, testset)
print(f'Creating dataloaders took {(t2-time.time())/60} mins')



# Calculate probabilities for masking of nodes if necessary
if cfg['use_masking'] or cfg['weighted_loss_var'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
    if isfile('node_label_vars.pt'):
        print('Using existing Node Label Variances for masking')
        mask_probs = torch.load('node_label_vars.pt')
    else:
        print('No node label variance file found\nCalculating Node Variances for Masking')
        mask_probs = calc_mask_probs(trainloader)
        torch.save(mask_probs, 'node_label_vars.pt')
else:
    #Masks are set to one in case it is wrongly used somewhere (when set to 1 masking results in multiplication with 1)
    mask_probs = torch.zeros(2000)+1


# getting feature and target sizes
num_features = trainset.__getitem__(0).x.shape[1]
num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
num_targets = 1


# choosing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #device = "cuda:0"
print(device)


# setting seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
numpy_seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Runs study if set in configuration file
if cfg["study::run"]:
    model = run_study(cfg, trainloader, testloader, device, mask_probs, num_features, num_edge_features)

#Runs crossvalidation
elif cfg['crossvalidation']:
    model = run_crossval(cfg, device, mask_probs, num_features, num_edge_features)

#Runs a single configuration    
else:
    model = run_single(cfg, trainloader, testloader, device, mask_probs, num_features, num_edge_features)



end = time.time()
logging.info(f'\nOverall Runtime: {(end-start)/60} min')
