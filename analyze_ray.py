import ray
import ray.tune as tune


import matplotlib.pyplot as plt

import numpy as np
import os
from training.training import objective
import torch

#control variables
name = 'GINE_GR' #Name tag added to the plots and their filenames
TEMP_DIR = '/home/tohlinger/RAY_TMP2/'
path = '/home/tohlinger/LEO/Running/GINE_ANGF_Y_GR/results/'


#loss and R2 Plots
i = 0
for file in os.listdir(path):
    if file.startswith('train_losses'):
        train_losses    = torch.load(path+file)
        test_losses     = torch.load(path+file.replace('train','test'))
        
        fig = plt.figure(i)
        ax = plt.gca()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(test_losses, label='Test Loss')
        ax.legend()
        fig.savefig(path+file.replace('train_losses','loss').replace('.pt','.png'), bbox_inches='tight')
        plt.close()


#RAY ANALYSIS
#result variables
ray.init(_temp_dir=TEMP_DIR,include_dashboard=False, num_cpus=1)
i_file = 0
offset = 0
unusable_trials = 0
usable_trials = 0
experiments_evaluated = 0
fastest_time = 9999999
slowest_time = 0
best_R2 = np.NINF
fastest_result = 0
slowest_result = 0


for file in os.listdir(path):
    if file.startswith('objective'):

        experiments_evaluated += 1



        tuner = tune.Tuner.restore(path+file, objective)
        result_grid=tuner.get_results()
        print(result_grid)
        
        N_trials = len(result_grid)

        
        if i_file == 0:
            N_params = 0
            params = {}
            for i in range(N_trials):
                if not result_grid[i].error and 'test_R2' in result_grid[i].metrics.keys():
                        for key in result_grid[i].config.keys():
                            params[key] = np.zeros(N_trials)
                            N_params += 1
                        break
            
            metrics = {
                'train_loss' :  np.zeros(N_trials),
                'test_loss' :   np.zeros(N_trials),
                'train_R2' :    np.zeros(N_trials),
                'test_R2' :     np.zeros(N_trials),
                'time_total_s'  :   np.zeros(N_trials)
                }
        else:
            for key in metrics.keys():
                metrics[key] = np.append(metrics[key],np.zeros(N_trials))
            for key in params.keys():
                params[key] = np.append(params[key], np.zeros(N_trials))
        
        
        #print(result_grid.get_best_result())
        for i in range(N_trials):
            print(i)
            print(result_grid[i].error)
            #print(result_grid[i].metrics['r2'])
            if not result_grid[i].error:
                if 'test_R2' in result_grid[i].metrics.keys():
                    if not torch.isnan(result_grid[i].metrics['test_R2']):
                        usable_trials += 1
                        for key in result_grid[i].config.keys():
                            if key in ['num_layers', 'hidden_size', 'embedding_dim', 'walk_length', 'reghead_size', 'reghead_layers', 'K', 'num_heads',
                                    'loss_type', 'use_batchnorm', 'use_masking', 'use_skipcon']:
                                params[key][i+offset] = int(result_grid[i].config[key])
                            elif key == 'gradclip' and result_grid[i].config[key] < 0.02:                            
                                params[key][i+offset] = 0
                            else:
                                params[key][i+offset] = result_grid[i].config[key]
                        for key in metrics.keys():
                            metrics[key][i+offset] = result_grid[i].metrics[key]

                        #Find best result, fastest result and slowest result
                        if 'test_R2' in result_grid[i].metrics.keys() and not torch.isnan(result_grid[i].metrics['test_R2']):
                            if result_grid[i].metrics['test_R2'] > best_R2:
                                best_result = result_grid[i]
                                best_R2 = result_grid[i].metrics['test_R2']
                            
                        if result_grid[i].metrics['time_total_s'] < fastest_time and result_grid[i].metrics['time_total_s'] != 0:
                            fastest_result = result_grid[i]
                            fastest = result_grid[i].metrics['time_total_s']
                            
                        if result_grid[i].metrics['time_total_s'] > slowest_time and result_grid[i].metrics['time_total_s'] != 0:
                            slowest_result = result_grid[i]
                            slowest = result_grid[i].metrics['time_total_s']
                    else:
                        print(f'Skipped {i} because Test R2 is nan')    
                        unusable_trials += 1    
                        
                else: 
                    print(f'Skipped {i} because Test R2 not in Metrics')
                    unusable_trials += 1
            else: 
                print(f'Skipped {i} because of Error')
                unusable_trials += 1
 
        offset = i+offset
        i_file += 1


        
print(f'{experiments_evaluated} experiments evaluated/n')
print(f'{usable_trials} trials evaluated')
print(f'{unusable_trials} trials unusable (Error)')
#print(result_grid[np.argmax(metrics['test_R2'])].config)
#print(result_grid[np.argmax(metrics['test_R2'])].metrics)
print('Best Result:/n')
print(best_result.config)
print(best_result.metrics)

print('Slowest Result:/n')
print(slowest_result.config)
print(slowest_result.metrics)

print('Fastest Result:/n')
print(fastest_result.config)
print(fastest_result.metrics)




#PLOTTING
#fig, axs = plt.subplots(N_params)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
i = 0
for key in params.keys():
    fig = plt.figure(i)        
    ax = plt.gca()
    if key == 'LR':
        ax.set_xscale('log')
        ax.scatter(10**params[key],metrics['test_R2'])
    elif key == 'loss_weight':
        ax.scatter(params[key],metrics['test_R2'], c=params['loss_type'])
    elif key == 'mask_bias':
        ax.scatter(params[key],metrics['test_R2'], c=params['use_masking'])
    else:
        ax.scatter(params[key],metrics['test_R2'])
    ax.set_title(name)
    ax.set_xlabel(key)
    ax.set_ylabel('Test R2')
    fig.savefig(key + name + ".png", bbox_inches='tight')
    i += 1
    plt.close()
    
#3D Plot of layers and HF
if 'num_layers' in params.keys() and 'hidden_size' in params.keys():
    fig = plt.figure(i+1)        
    ax = fig.add_subplot(projection='3d')
    ax.scatter(params['num_layers'], params['hidden_size'], metrics['test_R2'], c=metrics['test_R2'])
    ax.set_title(name)
    ax.set_xlabel('num_layers')
    ax.set_ylabel('hidden_size')
    ax.set_zlabel('Train R2')
    fig.savefig('Layers_HF_R2' + name + ".png", bbox_inches='tight')
    
#History Plot (R2 vs trials)
fig = plt.figure(i+2)
ax = fig.add_subplot()
ax.scatter(range(len(metrics['test_R2'])), metrics['test_R2'])
ax.set_title(name)
ax.set_xlabel('Trial')
ax.set_ylabel('Test R2')
fig.savefig('history_plot_' + name + '.png', bbox_inches='tight')





    
    


"""
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
"""