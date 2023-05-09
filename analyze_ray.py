import ray
import ray.tune as tune
import training.engine
import utils.utils
import training.training
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from training.training import objective

ray.init()
path = '/p/tmp/tobiasoh/machine_learning/ajusted_subset_alllog/results/objective_2023-05-07_10-30-45/'
tuner = tune.Tuner.restore(path, objective)
result_grid=tuner.get_results()
R2 = []
DM = []

HF = []
layers = []
LR = []
dropout = []
gc = []
skipcon = []
RHS = []
RHL = []
mask_bias = []
loss_weight = []
#print(result_grid.get_best_result())
for i in range(26):
    print(i)
    print(result_grid[i].error)
    #print(result_grid[i].metrics['r2'])
    if not result_grid[i].error:
        if 'r2' in result_grid[i].metrics.keys():
            if result_grid[i].metrics['r2'] < -1e33:
                continue
            DM.append(result_grid[i].metrics['discrete_measure'])
            R2.append(result_grid[i].metrics['r2'])
            with open(result_grid[i].log_dir/'params.json', 'r') as f:
                data = json.load(f)
            LR.append(data['LR'])
            print(type(data['LR']))
    
            layers.append(int(data['layers']))
            HF.append(int(data['HF']))
            gc.append(data['gradclip'])
            dropout.append(data['dropout'])
            skipcon.append(int(data['use_skipcon']))
            RHS.append(int(data['reghead_size']))
            RHL.append(int(data['reghead_layers']))
            mask_bias.append(data['mask_bias'])
            loss_weight.append(data['loss_weight'])
        else: print(f'Skipped i')

path = '/p/tmp/tobiasoh/machine_learning/ajusted_subset_alllog/results/objective_2023-05-08_08-27-20/'
tuner = tune.Tuner.restore(path, objective)
result_grid=tuner.get_results()
for i in range(26):
    print(i)
    print(result_grid[i].error)
    #print(result_grid[i].metrics['r2'])
    if not result_grid[i].error:
        if 'r2' in result_grid[i].metrics.keys():
            if result_grid[i].metrics['r2'] < -1e33:
                continue
            DM.append(result_grid[i].metrics['discrete_measure'])
            R2.append(result_grid[i].metrics['r2'])
            with open(result_grid[i].log_dir/'params.json', 'r') as f:
                data = json.load(f)
            LR.append(data['LR'])
            print(type(data['LR']))

            layers.append(int(data['layers']))
            HF.append(int(data['HF']))
            gc.append(data['gradclip'])
            dropout.append(data['dropout'])
            skipcon.append(int(data['use_skipcon']))
            RHS.append(int(data['reghead_size']))
            RHL.append(int(data['reghead_layers']))
            mask_bias.append(data['mask_bias'])
            loss_weight.append(data['loss_weight'])
        else: print(f'Skipped i')




print(HF)
print(R2)
print(DM)
print(gc)

name = 'lognorm'

fig1,ax1=plt.subplots()
ax1.scatter(gc,R2)
ax1.set_title("")
#ax1.set_xscale("log")
ax1.set_xlabel("Gradclip")
ax1.set_ylabel('R2')
fig1.savefig("GC_R2_25_" + name + ".png")

fig2,ax2=plt.subplots()
ax2.scatter(np.array(HF),np.array(R2))
ax2.set_title("")
ax2.set_xlabel("N Hidden Features")
ax2.set_ylabel('R2')
fig2.savefig("HF_R2_25_" + name + ".png")

fig3,ax3=plt.subplots()
ax3.scatter(LR,R2)
ax3.set_title("")
ax3.set_xscale("log");
ax3.set_xlabel("Learning Rate")
ax3.set_ylabel('R2')
fig3.savefig("LR_R2_25_" + name + ".png")

fig4,ax4=plt.subplots()
ax4.scatter(layers,R2)
ax4.set_title("")
ax4.set_xlabel("N Layers")
ax4.set_ylabel('R2')
fig4.savefig("layers_R2_25_" + name + ".png")

fig5,ax5=plt.subplots()
ax5.scatter(skipcon,R2)
ax5.set_title("")
ax5.set_xlabel("use_skipcon")
ax5.set_ylabel('R2')
fig5.savefig("skipcon_R2_25_" + name + ".png")

fig7,ax7=plt.subplots()
ax7.scatter(dropout,R2)
ax7.set_title("")
ax7.set_xlabel("dropout")
ax7.set_ylabel('R2')
fig7.savefig("dropout_cR2_25_" + name + ".png")

fig9,ax9=plt.subplots()
ax9.scatter(RHS,R2)
ax9.set_title("")
ax9.set_xlabel("Reghead Hidden Features")
ax9.set_ylabel('R2')
fig9.savefig("RHsize_R2_25_" + name + ".png")

fig8,ax8=plt.subplots()
ax8.scatter(RHL,R2)
ax8.set_title("")
ax8.set_xlabel("RegHead Layers")
ax8.set_ylabel('R2')
fig8.savefig("RHL_R2_25_" + name + ".png")


fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(dropout,layers, R2,c =R2)
ax6.set_title("")
ax6.set_xlabel("dropout")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("dropout_layers_R2_25_" + name + ".png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(gc,HF, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("Gradclip Norm")
ax6.set_ylabel('N HF')
ax6.set_zlabel('R2')
fig6.savefig("GC_HF_R2_25_" + name + ".png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(HF,layers, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N HF")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("HF_layers_R2_25_" + name + ".png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(RHL,RHS, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N Reghead Layers")
ax6.set_ylabel('N Reghead Features')
ax6.set_zlabel('R2')
fig6.savefig("RHL_RHS_R2_25_" + name + ".png")
