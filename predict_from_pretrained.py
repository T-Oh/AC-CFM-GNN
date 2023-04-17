# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:11:58 2023

@author: tobia
"""

from models.gine import GINE
import torch
import numpy as np
import json
from datasets.dataset import create_datasets, create_loaders, calc_mask_probs
from numpy.random import seed as numpy_seed
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from training.engine import Engine
import matplotlib.pyplot as plt

#Loading configuration
#PATH = "/p/tmp/tobiasoh/machine_learning/results/GINE/4000/maskingtests/50mask/lossmask/GINE.pt"
PATH = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GINE/4000/New_R2/Rescaled_Masking/lossmask/GINE.pt'
with open("configurations/configurationEval.json", "r") as io:
    cfg = json.load(io)



trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None)
trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg

model =GINE()
#getting feature and target sizes
num_features = trainset.__getitem__(0).x.shape[1]
num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
num_targets = 1



#choosing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #TO device represents the 'device' on which a torch.tensor is placed (cpu or cuda) -> cuda uses gpus
#device = "cuda:0"
print(device)

#setting seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
numpy_seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#choosing criterion
criterion = torch.nn.MSELoss(reduction = 'mean')  #TO defines the loss
criterion.to(device)
if cfg['use_masking']:
    mask_probs = calc_mask_probs(trainloader)
else:
    mask_probs = torch.ones(2000)
params = {
     "num_layers"    : cfg['num_layers'],
     "hidden_size"   : cfg['hidden_size'],
     "dropout"       : 0.0,
     "dropout_temp"  : 1.0,
     "heads"         : cfg['num_heads'],
     "use_batchnorm" : False,
     "gradclip"      : cfg['gradclip'],
     "use_skipcon"   : False,
     "reghead_size"  : cfg['reghead_size'],
     "reghead_layers": cfg['reghead_layers'],
     "use_masking"   : True,
     "mask_probs"    : mask_probs,
     
     "num_features"  : num_features,
     "num_edge_features" : num_edge_features,
     "num_targets"   : num_targets
    }
 #Loading GNN model
model = get_model(cfg, params)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

model.to(device)

#Choosing optimizer
optimizer = get_optimizer(cfg, model)

#Initializing engine
engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"],task=cfg["task"])

evaluation, output, labels = engine.eval(trainloader)

output = output.reshape(2000, int(len(output)/2000))
labels = labels.reshape(2000, int(len(labels)/2000))
print(output.shape)
print(labels.shape)
mean_output = np.zeros(2000)
mean_labels = np.zeros(2000)
var_output = np.zeros(2000)
var_labels = np.zeros(2000)
for i in range(2000):
    mean_output[i] = output[i].mean()
    mean_labels[i] = labels[i].mean()
    var_output[i] = output[i].var()
    var_labels[i] = labels[i].var()

    

fig1,ax1=plt.subplots()
x_ticks = np.array(range(2000))
ax1.bar(x_ticks, mean_labels,label='Mean Labels')
ax1.bar(x_ticks, mean_output, label='Mean Output')
#ax1.set_ylim(-0.1,1)
ax1.set_title("Mean Load Shed at Nodes")
ax1.set_xlabel("Node ID")
ax1.set_ylabel('Load Shed in p.U.')
ax1.legend()
fig1.savefig('mean_output.png')

fig2,ax2=plt.subplots()
x_ticks = np.array(range(2000))
ax2.bar(x_ticks, var_labels, label='Label Variance')
ax2.bar(x_ticks, var_output, label='Output Variance')
#ax2.set_ylim(-0.1,1)
ax2.set_title("Load Shed Variance at Node")
ax2.set_xlabel("Node ID")
ax2.set_ylabel('Variance')
ax2.legend()
fig2.savefig('mean_output.png')

fig3,ax3=plt.subplots()
x_ticks = np.array(range(2000))
test=ax3.scatter(mean_output, mean_labels,c=var_labels)
#ax1.bar(x_ticks, mean_output, label='Mean Output')
#ax1.set_ylim(-0.1,1)
ax3.set_title("Color = Label Variance")
ax3.set_xlabel("Mean Output")
ax3.set_ylabel('Mean Label')
ax3.legend()
fig3.colorbar(test)
fig3.savefig('mean_output_vs_mean_labels.png')


        
print(evaluation)
 
