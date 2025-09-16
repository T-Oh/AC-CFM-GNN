import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Model folders and display names
MODEL_FOLDERS = [
    '/home/tohlinger/PIK/Results/GTSF_StateReg/TAG/sl_50/paper_rerun/results/',
    '/home/tohlinger/PIK/Results/GTSF_StateReg/GAT/sl_50/paper_rerun/results/',
    '/home/tohlinger/PIK/Results/LTSF/sl_50/paper_rerun/results/'
]
MODEL_NAMES = ['TAG-LSTM', 'GAT-LSTM', 'LSTM']
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
METRICS_TO_PLOT = ['node_loss', 'edge_loss', 'loss']
YLIMS = {
    'node_loss': (0, 0.1),
    'edge_loss': (0, 7),
    'loss': (0, 7)
}

# Additional zoomed plot settings
ZOOMED_YLIMS = {
    'node_loss': (0, 0.0053),
    'edge_loss': (0, 2.2),
    'loss': (0, 2.2)
}

# Output directory
OUTPUT_DIR = '/home/tohlinger/HUI/Documents/Results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store metrics
all_model_metrics = {metric: {'train': [], 'test': []} for metric in METRICS_TO_PLOT}

# Load data from result files
for folder_idx, folder in enumerate(MODEL_FOLDERS):
    result_files = glob.glob(os.path.join(folder, 'results_*.pkl'))
    assert result_files, f"No result files found in {folder}"
    
    with open(result_files[0], 'rb') as f:
        result = pickle.load(f)

    for metric in METRICS_TO_PLOT:
        train_key = f'train_{metric}'
        test_key = f'test_{metric}'
        if train_key in result and test_key in result:
            all_model_metrics[metric]['train'].append(result[train_key])
            all_model_metrics[metric]['test'].append(result[test_key])
        else:
            print(f"Warning: Missing {metric} in {folder}")



# --- Combined plot with right-side legend ---
fig, axes = plt.subplots(nrows=len(METRICS_TO_PLOT), ncols=1, figsize=(6.5, 4.5), sharex=True, dpi=600)

for idx, metric in enumerate(METRICS_TO_PLOT):
    ax = axes[idx]
    for model_idx, model_name in enumerate(MODEL_NAMES):
        if model_idx < len(all_model_metrics[metric]['train']):
            train_curve = all_model_metrics[metric]['train'][model_idx]
            test_curve = all_model_metrics[metric]['test'][model_idx]
            ax.plot(train_curve, label=f'{model_name} Train', linestyle='-', color=COLORS[model_idx])
            ax.plot(test_curve, label=f'{model_name} Test', linestyle='--', color=COLORS[model_idx])
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
    ax.grid(True)
    if metric in YLIMS:
        ax.set_ylim(YLIMS[metric])
    ax.set_title(metric.replace('_', ' ').title(), fontsize=10)

axes[-1].set_xlabel('Epoch', fontsize=9)

# Put legend to the right of the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='center left', bbox_to_anchor=(0.8, 0.5),
    fontsize=8, frameon=False
)

fig.subplots_adjust(right=0.8, hspace=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_comparison_combined_rightlegend.png'), bbox_inches='tight')
plt.close()

# --- Zoomed-in combined plot ---
fig, axes = plt.subplots(nrows=len(METRICS_TO_PLOT), ncols=1, figsize=(6.5, 4.5), sharex=True, dpi=600)

for idx, metric in enumerate(METRICS_TO_PLOT):
    ax = axes[idx]
    for model_idx, model_name in enumerate(MODEL_NAMES):
        if model_idx < len(all_model_metrics[metric]['train']):
            train_curve = all_model_metrics[metric]['train'][model_idx]
            test_curve = all_model_metrics[metric]['test'][model_idx]
            ax.plot(train_curve, label=f'{model_name} Train', linestyle='-', color=COLORS[model_idx])
            ax.plot(test_curve, label=f'{model_name} Test', linestyle='--', color=COLORS[model_idx])
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
    ax.grid(True)
    if metric in ZOOMED_YLIMS:
        ax.set_ylim(ZOOMED_YLIMS[metric])
    ax.set_title(metric.replace('_', ' ').title() + ' (Zoomed)', fontsize=10)

axes[-1].set_xlabel('Epoch', fontsize=9)

# Legend to the right again
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc='center left', bbox_to_anchor=(0.8, 0.5),
    fontsize=8, frameon=False
)

fig.subplots_adjust(right=0.8, hspace=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_comparison_combined_zoomed.png'), bbox_inches='tight')
plt.close()



# --- Individual plots ---
for metric in METRICS_TO_PLOT:
    plt.figure(figsize=(5, 3), dpi=600)
    for model_idx, model_name in enumerate(MODEL_NAMES):
        if model_idx < len(all_model_metrics[metric]['train']):
            train_curve = all_model_metrics[metric]['train'][model_idx]
            test_curve = all_model_metrics[metric]['test'][model_idx]
            plt.plot(train_curve, label=f'{model_name} Train', linestyle='-', color=COLORS[model_idx])
            plt.plot(test_curve, label=f'{model_name} Test', linestyle='--', color=COLORS[model_idx])
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(metric.replace('_', ' ').title())
    plt.grid(True)
    if metric in YLIMS:
        plt.ylim(YLIMS[metric])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{metric}_comparison.png'), bbox_inches='tight')
    plt.close()
