from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import plotly.express as px
import json


# for colors pip install mycolorpy
from mycolorpy import colorlist as mcp


import pandas as pd

# parameter_keys = ["optim::LR", "train_set::batchsize"]
# parameter_keys = ["ARMA::num_internal_layers_0", "ARMA::num_internal_layers_1", "ARMA::num_internal_stacks_0"]
parameter_keys = ["ARMA::num_internal_layers", "ARMA::num_internal_stacks"]


result_keys = ["train_loss", "train_acc", "train_R2", "valid_loss",
               "valid_acc", "valid_R2", "test_loss", "test_acc", "test_R2"]

result_keys = ["test_acc", "test_R2"]


legendVar = "test_R2"
numLegendCategories=15

result_modes = {}
result_modes["train_loss"] = "min"
result_modes["valid_loss"] = "min"
result_modes["test_loss"] = "min"

result_modes["train_acc"] = "max"
result_modes["valid_acc"] = "max"
result_modes["test_acc"] = "max"

result_modes["train_R2"] = "max"
result_modes["valid_R2"] = "max"
result_modes["test_R2"] = "max"


path_runs = "/home/chris/work/pik/projects/dataset_gnn/N020/ray_arma/run002/ArmaNet2_lr_bs"

listDirectories = []
for file in sorted(os.listdir(path_runs)):
    if "NN_tune_trainable" in file:
        listDirectories.append(file)


numRuns = len(listDirectories)
params = {}

def getIdx(word, character):
    try:
        return word.index(character)
    except:
        return -1



def get_result_value(result, mode):
    if mode == "min":
        return min(result)
    if mode == "max":
        return max(result) 

def read_input_data(path_runs,numRuns):
    results_dataframe = pd.DataFrame(
        columns=parameter_keys, index=np.arange(numRuns))

    for i in range(numRuns):
        parameter_file_name = Path(path_runs + '/' + listDirectories[i] + '/params.json')
        with open(parameter_file_name) as f:
            parameter_json = json.load(f)
        for j in range(len(parameter_keys)):
            vName = parameter_keys[j]
            parameters_loaded = parameter_json[vName]
            if type(parameters_loaded) == list:
                num_parameters_loades = len(parameters_loaded)
                for k in range(num_parameters_loades):
                    name_dataframe = vName + '_' + str(k)
                    results_dataframe.at[i, name_dataframe] = parameters_loaded[k]        
            else:
                results_dataframe.at[i, vName] = parameter_json[vName]
        result_file_name = Path(
            path_runs + '/' + listDirectories[i] + '/progress.csv')
        result_file = pd.read_csv(result_file_name)
        for j in range(len(result_keys)):
            vName = result_keys[j]
            mode = result_modes[vName]
            results_dataframe.at[i, vName] = get_result_value(result_file[vName], mode)
    return results_dataframe



def make_linspace(values):
    min_value = min(values)
    max_value = max(values)
    if min_value < 0:
        min_value = 1.1* min_value
    else:
        min_value = .9 * min_value
    
    if max_value > 0:
        max_value = 1.1 * max_value
    else:
        max_value = .9 * max_value
    return np.linspace(min_value, max_value, num=numLegendCategories)

def prepare_data(df):
    arange_list = make_linspace(df[legendVar])
    df["legendVar"] = pd.cut(df[legendVar], arange_list)
    cols = df.columns.to_list()
    cols.remove("legendVar")
    x = [i for i, _ in enumerate(cols)]
#     colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']
    colours = mcp.gen_color(cmap="plasma",n=numLegendCategories)
    # create dict of categories: colours
    colours = {df["legendVar"].cat.categories[i]: colours[i]
            for i, _ in enumerate(df["legendVar"].cat.categories)}
    return x, cols, colours

def plot_data(fix, axes, x, cols, df, min_max_range):
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            legendVar_category = df.loc[idx, "legendVar"]
            ax.plot(x, df.loc[idx, cols], colours[legendVar_category])
            # ax.plot(x, df.loc[idx, cols])
        ax.set_xlim([x[i], x[i+1]])


# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, ticks):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    step = val_range / float(ticks-1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)
    


df = read_input_data(path_runs,numRuns)
x, cols, colours = prepare_data(df)
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(25, 10))
min_max_range = {}
plot_data(fig,axes, x, cols, df, min_max_range)

# plotting

for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]])


# Move the final axis' ticks to the right-hand side
ax = plt.twinx(axes[-1])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]])


# Remove space between subplots
plt.subplots_adjust(wspace=0)

# Add legend to plot
plt.legend(
    [plt.Line2D((0, 1), (0, 0), color=colours[cat])
     for cat in df["legendVar"].cat.categories],
    df["legendVar"].cat.categories,
    bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

plt.title("HParams")

fig.show()


print("finished")
