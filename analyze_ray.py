import ray
import ray.tune as tune
import training.engine
import utils.utils
import training.training
import json
import matplotlib.pyplot as plt

ray.init()
path = '/p/tmp/tobiasoh/machine_learning/NodeRegression2/results/objective_2023-03-10_15-31-13'
tuner = tune.Tuner.restore(path)
result_grid=tuner.get_results()
R2 = []
DM = []
HF = []
layers = []
LR = []
heads = []
dropout = []
print(result_grid[0])
for i in range(44):
    print(i)
    if i == 45:
        continue
    if not result_grid[i].error:
        if result_grid[i].metrics['r2'] < -1e33:
            continue
        DM.append(result_grid[i].metrics['discrete_measure'])
        R2.append(result_grid[i].metrics['r2'])
        with open(result_grid[i].log_dir/'params.json', 'r') as f:
           data = json.load(f)
        LR.append(data['LR'])
        heads.append(data['heads'])
        layers.append(data['layers'])
        HF.append(data['HF'])
        dropout.append(data['dropout'])
print(HF)
print(R2)
print(DM)

fig1,ax1=plt.subplots()
ax1.scatter(HF,R2)
ax1.set_title("")
ax1.set_xlabel("N Hidden Features")
ax1.set_ylabel('R2')
fig1.savefig("HF_R2_75_1500.png")

fig2,ax2=plt.subplots()
ax2.scatter(HF,DM)
ax2.set_title("")
ax2.set_xlabel("N Hidden Features")
ax2.set_ylabel('Discrete Measure')
fig2.savefig("HF_DM_75_1500.png")

fig3,ax3=plt.subplots()
ax3.scatter(LR,DM)
ax3.set_title("")
ax3.set_xscale("log");
ax3.set_xlabel("Learning Rate")
ax3.set_ylabel('DM')
fig3.savefig("LR_DM_75_1500.png")

fig4,ax4=plt.subplots()
ax4.scatter(layers,DM)
ax4.set_title("")
ax4.set_xlabel("N Layers")
ax4.set_ylabel('DM')
fig4.savefig("layers_DM_75_1500.png")

fig5,ax5=plt.subplots()
ax5.scatter(heads,DM)
ax5.set_title("")
ax5.set_xlabel("N Heads")
ax5.set_ylabel('DM')
fig5.savefig("heads_DM_75_1500.png")

fig7,ax7=plt.subplots()
ax7.scatter(dropout,layers,c=DM)
ax7.set_title("")
ax7.set_xlabel("Dropout")
ax7.set_ylabel('DM')
fig7.savefig("dropout_layers_cDM_75_1500.png")




fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(heads,layers, R2,c =R2)
ax6.set_title("")
ax6.set_xlabel("N Heads")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("heads_layers_R2_75_1500.png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(heads,HF, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N Heads")
ax6.set_ylabel('N HF')
ax6.set_zlabel('R2')
fig6.savefig("heads_HF_R2_75_1500.png")

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.scatter(HF,layers, R2,c=R2)
ax6.set_title("")
ax6.set_xlabel("N HF")
ax6.set_ylabel('N Layers')
ax6.set_zlabel('R2')
fig6.savefig("HF_layers_R2_75_1500.png")
