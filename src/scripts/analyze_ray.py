import ray
import ray.tune as tune


import matplotlib.pyplot as plt

import numpy as np
import os
from training.training import objective
import torch
import glob
import pickle

#control variables
name = 'Test2' #Name tag added to the plots and their filenames
TEMP_DIR = '/home/tohlinger/RAY_TMP2/'
path = '/home/tohlinger/HUI/Documents/hi-accf-ml/results/'
TASK = 'NR'

def plot_result_files(path):
    # Get all result files matching the pattern 'results_*.pkl'
    result_files = glob.glob(path+'/results_*.pkl')

    # Dictionary to store optimal values for each metric across all files
    optimal_metrics = {}

    # Iterate over each results file
    for result_file in result_files:
        # Load the result dictionary from the pickle file
        with open(result_file, 'rb') as f:
            result = pickle.load(f)

        # Filter the result dictionary for metric names that start with 'train_'
        train_metrics = {k: v for k, v in result.items() if k.startswith('train_')}

        # Iterate over each training metric
        for train_metric_name, train_values in train_metrics.items():
            # Derive the corresponding test metric by replacing 'train_' with 'test_'
            test_metric_name = train_metric_name.replace('train_', 'test_')
            
            # Check if the corresponding test metric exists
            if test_metric_name in result:
                test_values = result[test_metric_name]

                # Get the actual metric name without 'train_' or 'test_' prefix
                metric_name = train_metric_name.replace('train_', '')

                # Initialize optimal metric list if not already present
                if metric_name not in optimal_metrics:
                    optimal_metrics[metric_name] = []

                # Check if the metric is multiclass (a list of lists)
                if isinstance(train_values[0], list):
                    # Prepare a figure for plotting all class curves
                    plt.figure()

                    # Iterate through the classes
                    optim_values = []
                    for class_idx in range(len(train_values[0])):
                        # Extract the metric values for the specific class across epochs
                        train_class_values = [epoch[class_idx] for epoch in train_values]
                        test_class_values = [epoch[class_idx] for epoch in test_values]

                        # Plot the class-specific train and test values
                        plt.plot(train_class_values, label=f'Train Class {class_idx}')
                        plt.plot(test_class_values, label=f'Test Class {class_idx}', linestyle='--')

                        # Get the optimal values (min for loss, max for accuracy-like metrics)
                        if 'loss' in metric_name.lower():
                            optimal_test_value = np.min(test_class_values)
                        else:
                            optimal_test_value = np.max(test_class_values)
                        optim_values.append(optimal_test_value)

                    # Store the optimal value for each class
                    optimal_metrics[metric_name].append([optim_values])

                    # Finalize the plot
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                    plt.title(f'{metric_name} for all classes')
                    plt.legend()

                    # Save the plot with the filename indicating the result file
                    result_filename = os.path.splitext(os.path.basename(result_file))[0]
                    plot_filename = f'plots/{result_filename}_{metric_name}_all_classes.png'
                    plt.savefig(plot_filename)
                    plt.close()
                else:
                    # Plot the metric for a single list of values (non-multiclass metric)
                    plt.figure()
                    plt.plot(train_values, label='Train')
                    plt.plot(test_values, label='Test')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                    plt.ylim((0,1))
                    plt.title(f'{metric_name}')
                    plt.legend()

                    # Save the plot with the filename indicating the result file
                    result_filename = os.path.splitext(os.path.basename(result_file))[0]
                    plot_filename = f'plots/{result_filename}_{metric_name}_zoom.png'
                    plt.savefig(plot_filename)
                    plt.close()

                    # Get the optimal values (min for loss, max for accuracy-like metrics)
                    if 'loss' in metric_name.lower():
                        optimal_test_value = np.min(test_values)
                    else:
                        optimal_test_value = np.max(test_values)

                    # Store the optimal value
                    optimal_metrics[metric_name].append(optimal_test_value)

    # Calculate and print the mean of the optimal values for each metric
    for metric_name, optimal_values in optimal_metrics.items():
        # If the metric has multiple classes, we need to reshape the optimal values
        if isinstance(optimal_values[0], list):
            optimal_values = np.array(optimal_values)

        # Print the mean optimal value per class
        mean_optimal_value = np.mean(optimal_values, axis=0)
        print(f'Mean optimal {metric_name}: {mean_optimal_value}')

# Ensure the directory exists for saving plots
os.makedirs('plots', exist_ok=True)

# Call the function
plot_result_files(path)


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
best_loss = np.inf
fastest_result = 0
slowest_result = 0


for file in os.listdir(path):
    if file.startswith(name):

        experiments_evaluated += 1


        tuner = tune.Tuner.restore(path+file, objective)
        result_grid=tuner.get_results()
        print(result_grid)
        
        N_trials = len(result_grid)

        
        if i_file == 0:
            N_params = 0
            params = {}
            for i in range(N_trials):
                if not result_grid[i].error and (TASK == 'GC' or 'test_R2' in result_grid[i].metrics.keys()):
                        for key in result_grid[i].config.keys():
                            params[key] = np.zeros(N_trials)
                            N_params += 1
                        break
            
            metrics = {
                'train_loss' :  np.zeros(N_trials),
                'test_loss' :   np.zeros(N_trials),
                'train_R2' :    np.zeros(N_trials),
                'train_R2_2' :  np.zeros(N_trials),
                'test_R2' :     np.zeros(N_trials),
                'test_R2_2' :     np.zeros(N_trials),
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
                if TASK == 'GC' or ('test_R2' in result_grid[i].metrics.keys() and not torch.isnan(torch.tensor(result_grid[i].metrics['test_R2']))):
                    
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
                    if TASK == 'GC':
                        if result_grid[i].metrics['test_loss'] < best_loss:
                                best_result = result_grid[i]
                                best_loss = result_grid[i].metrics['test_loss']
                    else:
                        if 'test_R2' in result_grid[i].metrics.keys() and not torch.isnan(torch.tensor(result_grid[i].metrics['test_R2'])):
                            if result_grid[i].metrics['test_R2'] > best_R2:
                                best_result = result_grid[i]
                                best_R2 = result_grid[i].metrics['test_R2']
                        
                    if result_grid[i].metrics['time_total_s'] < fastest_time and result_grid[i].metrics['time_total_s'] != 0:
                        fastest_result = result_grid[i]
                        fastest = result_grid[i].metrics['time_total_s']
                        
                    if result_grid[i].metrics['time_total_s'] > slowest_time and result_grid[i].metrics['time_total_s'] != 0:
                        slowest_result = result_grid[i]
                        slowest = result_grid[i].metrics['time_total_s']
                elif 'R2' not in result_grid[i].metrics.keys():
                    print(f'Skipped {i} because Test R2 not in Metrics') 
                    unusable_trials += 1    
                        
                else: 
                    print(f'Skipped {i} because Test R2 is nan')                    
                    unusable_trials += 1
            else: 
                print(f'Skipped {i} because of Error')
                unusable_trials += 1
 
        offset = i+offset
        i_file += 1


        
print(f'{experiments_evaluated} experiments evaluated/n')
print(f'{usable_trials} trials evaluated')
print(f'{unusable_trials} trials unusable (Error)')
#print(result_grid[np.argmax(metrics['R2'])].config)
#print(result_grid[np.argmax(metrics['R2'])].metrics)
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
if TASK == 'GC':    METRIC = metrics['test_loss']
else:               METRIC = metrics['test_R2']
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
i = 0
for key in params.keys():
    fig = plt.figure(i)        
    ax = plt.gca()
    if key == 'LR':
        ax.set_xscale('log')
        ax.scatter(10**params[key],METRIC)
    elif key == 'loss_weight':
        ax.scatter(params[key],METRIC, c=params['loss_type'])
    elif key == 'mask_bias':
        ax.scatter(params[key],METRIC, c=params['use_masking'])
    else:
        ax.scatter(params[key],METRIC)
    ax.set_title(name)
    ax.set_xlabel(key)
    ax.set_ylabel('Test R2')
    fig.savefig('plots/'+key + name + ".png", bbox_inches='tight')
    i += 1
    plt.close()

    #Zoomed plots
    fig = plt.figure(i)        
    ax = plt.gca()
    if key == 'LR':
        ax.set_xscale('log')
        ax.scatter(10**params[key],METRIC)
    elif key == 'loss_weight':
        ax.scatter(params[key],METRIC, c=params['loss_type'])
    elif key == 'mask_bias':
        ax.scatter(params[key],METRIC, c=params['use_masking'])
    else:
        ax.scatter(params[key],METRIC)
    ax.set_ylim(-1,1)
    ax.set_title(name)
    ax.set_xlabel(key)
    ax.set_ylabel('Test R2')
    fig.savefig('plots/'+key + name + ".png", bbox_inches='tight')
    i += 1
    plt.close()
    
#3D Plot of layers and HF
if 'num_layers' in params.keys() and 'hidden_size' in params.keys():
    fig = plt.figure(i+1)        
    ax = fig.add_subplot(projection='3d')
    ax.scatter(params['num_layers'], params['hidden_size'], METRIC, c=METRIC)
    ax.set_title(name)
    ax.set_xlabel('num_layers')
    ax.set_ylabel('hidden_size')
    ax.set_zlabel('Train R2')
    fig.savefig('plots/'+'Layers_HF_R2' + name + ".png", bbox_inches='tight')
    
#History Plot (R2 vs trials)
fig = plt.figure(i+2)
ax = fig.add_subplot()
ax.scatter(range(len(METRIC)), METRIC)
ax.set_title(name)
ax.set_xlabel('Trial')
ax.set_ylabel('Test R2')
fig.savefig('plots/'+'history_plot_' + name + '.png', bbox_inches='tight')

#Zoomed history plot
fig = plt.figure(i+2)
ax = fig.add_subplot()
ax.scatter(range(len(METRIC)), METRIC)
ax.set_ylim(-1,1)
ax.set_title(name)
ax.set_xlabel('Trial')
ax.set_ylabel('Test R2')
fig.savefig('plots/'+'history_plot_' + name + '.png', bbox_inches='tight')





    
    


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
