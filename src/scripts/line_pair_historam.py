import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

PATH_DATASET = 'data/damage_files/'


def get_initial_damages(file):
    '''
    returns the sorted initial damages of each scenario in damages [N_scenarios][step,line_id]
    where step is reassigned so that it starts with 0 and increments by 1 except if two lines 
    were destroyed in the same step
    '''
    
    #load scenario file which stores the initial damages
    f = open(file,'r')
    lines= f.readlines()
    damages = []
    for i in range(len(lines)):
        lines[i] = lines[i].replace("[", '')
        lines[i] = lines[i].replace(']', '')
        lines[i] = lines[i].replace('(', '')
        lines[i] = lines[i].replace(')', '')
        lines[i] = lines[i].replace('"', '')
        lines[i] = lines[i].replace(',', '')
        line = np.array(list(map(int, lines[i].split())))
        scenario_dmgs=np.reshape(line,(-1,2))
        scenario_dmgs=scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]
        #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
        index = 0
        for j in range(0,len(scenario_dmgs)-1):
            increment = 0
            if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
                increment = 1
            scenario_dmgs[j,0] = index
            index += increment
        scenario_dmgs[-1,0] = index
        damages.append(scenario_dmgs)
    return damages



pair_hist = {}
for file in os.listdir(PATH_DATASET):
    
    damages = get_initial_damages(PATH_DATASET+file)

    for i in range(len(damages)):
        if len(damages[i]) > 1:
            
            init_pair = (int(damages[i][0,1]), int(damages[i][1,1]))
            if init_pair in pair_hist:
                pair_hist[init_pair] += 1
            else:
                pair_hist[init_pair] = 1

print('There are ', len(pair_hist), 'unique initial pairs\n')

print(sorted(pair_hist.items(), key=lambda item: item[1], reverse=True)[:10])

opposite_hist = {}
for count in pair_hist.values():
    if count in opposite_hist:
        opposite_hist[count] += 1
    else:
        opposite_hist[count] = 1


counts = list(opposite_hist.keys())
num_pairs = list(opposite_hist.values())


plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300

fig0,ax0=plt.subplots()
ax0.bar(counts, num_pairs)
ax0.set_xlim(0,42)
ax0.set_xlabel('Number of Appearances')
ax0.set_ylabel('Number of Pairs')
fig0.savefig("line_pair_hist.png", bbox_inches='tight')


fig1,ax1=plt.subplots()
ax1.bar(counts, num_pairs)
ax1.set_ylim(0,1000)
ax1.set_xlim(0,42)
ax1.set_xlabel('Number of Appearances')
ax1.set_ylabel('Number of Pairs')
fig1.savefig("line_pair_hist_zoom.png", bbox_inches='tight')







                    












