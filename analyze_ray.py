import ray
import ray.tune as tune
import training.engine
import utils.utils
import training.training
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

ray.init()
path = '/p/tmp/tobiasoh/machine_learning/results/GINE/4000/study/objective_2023-03-28_13-29-27/'
tuner = tune.Tuner.restore(path)
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
print(result_grid.get_best_result())
for i in range(20):
    print(i)
    if not result_grid[i].error:
        if result_grid[i].metrics['r2'] < -1e33:
            continue
        DM.append(result_grid[i].metrics['discrete_measure'])
        R2.append(result_grid[i].metrics['r2'])
        with open(result_grid[i].log_dir/'params.json', 'r') as f:
           data = json.load(f)
        LR.append(data['LR'])
        print(type(data['LR']))
        
        layers.append(data['layers'])
        HF.append(data['HF'])
        gc.append(data['gradclip'])
        dropout.append(data['dropout'])
        skipcon.append(data['use_skipcon'])
        RHS.append(data['reghead_size'])
        RHL.append(data['reghead_layers'])

path = '/p/tmp/tobiasoh/machine_learning/NodeRegression/results/objective_2023-03-29_09-15-15/'
tuner = tune.Tuner.restore(path)
result_grid=tuner.get_results()
for i in range(20):
    print(i)
    if not result_grid[i].error:
        if result_grid[i].metrics['r2'] < -1e33:
            continue
        DM.append(result_grid[i].metrics['discrete_measure'])
        R2.append(result_grid[i].metrics['r2'])
        with open(result_grid[i].log_dir/'params.json', 'r') as f:
           data = json.load(f)
        LR.append(data['LR'])
        print(type(data['LR']))

        layers.append(data['layers'])
        HF.append(data['HF'])
        gc.append(data['gradclip'])
        dropout.append(data['dropout'])
        skipcon.append(data['use_skipcon'])
        RHS.append(data['reghead_size'])
        RHL.append(data['reghead_layers'])




print(HF)
print(R2)
print(DM)
print(gc)

fig1,ax1=plt.subplots()
ax1.scatter(gc,R2)
ax1.set_title("")
ax1.set_xlabel("Norm for gradient clipping")
ax1.set_ylabel('R2')
fig1.savefig("GC_R2_50_4000.png")

fig2,ax2=plt.subplots()
ax2.scatter(np.array(HF),np.array(DM))
ax2.set_title("")
ax2.set_xlabel("N Hidden Features")
ax2.set_ylabel('Discrete Measure')
fig2.savefig("HF_DM_50_4000.png")

fig3,ax3=plt.subplots()
ax3.scatter(LR,R2)
ax3.set_title("")
ax3.set_xscale("log");
ax3.set_xlabel("Learning Rate")
ax3.set_ylabel('R2')
fig3.savefig("LR_R2_50_4000.png")

fig4,ax4=plt.subplots()
ax4.scatter(layers,R2)
ax4.set_title("")
ax4.set_xlabel("N Layers")
ax4.set_ylabel('R2')
fig4.savefig("layers_R2_50_4000.png")

fig5,ax5=plt.subplots()
ax5.scatter(skipcon,R2)
ax5.set_title("")
ax5.set_xlabel("use_skipcon")
ax5.set_ylabel('R2')
fig5.savefig("skipcon_R2_50_4000.png")

fig7,ax7=plt.subplots()
ax7.scatter(dropout,layers,c=R2)
ax7.set_title("")
ax7.set_xlabel("dropout")
ax7.set_ylabel('R2')
fig7.savefig("dropout_layers_cR2_50_4000.png")

fig9,ax9=plt.subplots()
ax9.scatter(RHS,R2,c=RHL)
ax9.set_title("")
ax9.set_xlabel("Reghead Hidden Features")
ax9.set_ylabel('R2')
fig9.savefig("RHsize_R2_cRHL_750_4000.png")

fig8,ax8=plt.subplots()
ax8.scatter(RHL,R2,c=RHS)
ax8.set_title("")
ax8.set_xlabel("RegHead Layers")
ax8.set_ylabel('R2')
fig8.savefig("RHL_R2_cRHS_50_4000.png")


fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(dropout,layers, R2,c =R2)
ax6.set_title("")
ax6.set_xlabel("dropout")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("dropout_layers_R2_50_1500.png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(gc,HF, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N Heads")
ax6.set_ylabel('N HF')
ax6.set_zlabel('R2')
fig6.savefig("GC_HF_R2_50_1500.png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(HF,layers, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N HF")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("HF_layers_R2_50_1500.png")
