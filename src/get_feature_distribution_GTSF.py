import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

def load_data_from_scenario_folders(processed_path):
    """Generator to load data files from scenario folders."""
    for root, dirs, files in os.walk(processed_path):
        for file in files:
            if file.endswith(".pt") and "pre" not in file:
                yield torch.load(os.path.join(root, file))

def update_histogram(data, hist, bin_edges):
    """Updates the histogram counts."""
    new_hist, _ = np.histogram(data, bins=bin_edges)
    hist += new_hist
    return hist

def natural_sort_key(filename):
    """Key function for natural sorting, splitting text and numbers."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def compute_distributions(processed_path, num_bins=50):
    """Computes histograms for features and labels incrementally."""
    # Initialize histograms for features and labels
    x_hist = None
    node_labels_hist = None
    y_hist = np.zeros(num_bins)
    y_cumulative_hist = np.zeros(num_bins)
    bin_edges_x = None
    bin_edges_node_labels = None
    bin_edges_y = np.linspace(0, 67.1, num_bins + 1)  # Adjust bins as needed
    bin_edges_y_cumulative = np.linspace(0, 67.1, num_bins + 1)  # Adjust bins as needed

    # Include static data
    static_file = os.path.join(processed_path, "data_static.py")
    if os.path.exists(static_file):
        static_data = torch.load(static_file)
        if bin_edges_x is None:
            bin_edges_x = [np.histogram_bin_edges(static_data.x[:, i], bins=num_bins) for i in range(static_data.x.shape[1])]
        for i in range(static_data.x.shape[1]):
            x_hist[i] = update_histogram(static_data.x[:, i].numpy(), x_hist[i], bin_edges_x[i])

    # Loop through scenario data
    for root, dirs, files in os.walk(processed_path):
        # Natural sorting of files
        scenario_files = sorted(
            [f for f in files if f.endswith(".pt") and "pre" not in f],
            key=natural_sort_key
        )
        for i, file in enumerate(scenario_files):
            file_path = os.path.join(root, file)
            data = torch.load(file_path)
            
            if x_hist is None:
                # Initialize histograms for features and node labels
                bin_edges_x = [np.histogram_bin_edges(data.x[:, i], bins=num_bins) for i in range(data.x.shape[1])]
                x_hist = [np.zeros(len(bin_edges_x[i]) - 1) for i in range(data.x.shape[1])]
                bin_edges_node_labels = [np.histogram_bin_edges(data.node_labels[:, i], bins=num_bins) for i in range(data.node_labels.shape[1])]
                node_labels_hist = [np.zeros(len(bin_edges_node_labels[i]) - 1) for i in range(data.node_labels.shape[1])]

            # Update feature histograms
            for j in range(data.x.shape[1]):
                x_hist[j] = update_histogram(data.x[:, j].numpy(), x_hist[j], bin_edges_x[j])

            # Update node label histograms
            for j in range(data.node_labels.shape[1]):
                node_labels_hist[j] = update_histogram(data.node_labels[:, j].numpy(), node_labels_hist[j], bin_edges_node_labels[j])

            # Update scalar label histograms
            y_hist = update_histogram([data.y / 1000], y_hist, bin_edges_y)

            # Track the last file in the scenario
            if i == len(scenario_files) - 1:
                y_cumulative_hist = update_histogram([data.y_cummulative / 1000], y_cumulative_hist, bin_edges_y_cumulative)

    return x_hist, bin_edges_x, node_labels_hist, bin_edges_node_labels, y_hist, bin_edges_y, y_cumulative_hist, bin_edges_y_cumulative



def plot_histograms(hist, bin_edges, title, save_path, NAME):
    """Plots histograms and saves them."""
    lettering = {
        'Feature': {
            0: 'P_injection',
            1: 'Q_injection',
            2: 'V_real',
            3: 'V_imag',
            4: 'PQ_bus',
            5: 'PV_bus',
            6: 'Slack_bus',
            7: 'inactive_bus'
        },
        'Node_Label': {
            0: 'V_real',
            1: 'V_imag'
        },
        'Graph_label': {
            0: 'total_load_shed_step'
        },
        'Cummulative_Graph_Label': {
            0: 'total_load_shed_cummulative'
            }
    }
    for i, (h, b) in enumerate(zip(hist, bin_edges)):
        plt.bar(b[:-1], h, width=np.diff(b), align='edge', alpha=0.7)   
        plt.xlabel(lettering[title][i])
        plt.ylabel('Frequency')
        plt.title(lettering[title][i])
        plt.savefig(os.path.join(save_path, f'{NAME}_{title.lower().replace(" ", "_")}_{i+1}.png'))
        plt.clf()

if __name__ == "__main__":
    processed_path = "normalized/"
    save_dir = "feat_dists/"
    NAME = 'normalized'
    os.makedirs(save_dir, exist_ok=True)

    # Compute distributions incrementally
    results = compute_distributions(processed_path)
    print(results[6])
    print(results[7])

    # Plot and save histograms
    plot_histograms(results[0], results[1], "Feature", save_dir, NAME)
    plot_histograms(results[2], results[3], "Node_Label", save_dir, NAME)
    #plot_histograms(results[4], results[5], "Graph_Label", save_dir)
    #plot_histograms(results[6], results[7], "Cummulative_Graph_Label", save_dir)

    # Plot scalar labels
    plt.bar(results[5][:-1], results[4], width=np.diff(results[5]), align='edge', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('y Distribution')
    plt.savefig(os.path.join(save_dir, f'{NAME}_y_distribution.png'))
    plt.clf()

    plt.bar(results[7][:-1], results[6], width=np.diff(results[7]), align='edge', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('y_cumulative Distribution')
    plt.savefig(os.path.join(save_dir, f'{NAME}_y_cumulative_distribution.png'))
    plt.clf()
