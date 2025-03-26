import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data_from_scenario_folders(processed_path):
    """Generator to load data files from scenario folders."""
    for root, dirs, files in os.walk(processed_path):
        for file in files:
            if file.endswith(".pt") and "pre" not in file:
                yield torch.load(os.path.join(root, file))

def get_global_min_max(processed_path):
    """First pass: Determine global min and max for all features."""
    min_max = {
        "x": None, "node_labels": None, "edge_attr": None,
        "y": [float('inf'), float('-inf')], 
        "y_cumulative": [float('inf'), float('-inf')]
    }

    # Include static data
    static_file = os.path.join(processed_path, "data_static.py")
    if os.path.exists(static_file):
        static_data = torch.load(static_file)
        min_max["x"] = [
            (torch.min(static_data.x[:, i]).item(), torch.max(static_data.x[:, i]).item()) 
            for i in range(static_data.x.shape[1])
        ]

    # Scan through scenario data to update min and max
    for data in tqdm(load_data_from_scenario_folders(processed_path), desc="Scanning Min/Max"):
        # Features
        if min_max["x"] is None:
            min_max["x"] = [(torch.min(data.x[:, i]).item(), torch.max(data.x[:, i]).item()) for i in range(data.x.shape[1])]
        else:
            for i in range(data.x.shape[1]):
                min_val, max_val = torch.min(data.x[:, i]).item(), torch.max(data.x[:, i]).item()
                min_max["x"][i] = (min(min_max["x"][i][0], min_val), max(min_max["x"][i][1], max_val))

        # Node labels
        if min_max["node_labels"] is None:
            min_max["node_labels"] = [(torch.min(data.node_labels[:, i]).item(), torch.max(data.node_labels[:, i]).item()) for i in range(data.node_labels.shape[1])]
        else:
            for i in range(data.node_labels.shape[1]):
                min_val, max_val = torch.min(data.node_labels[:, i]).item(), torch.max(data.node_labels[:, i]).item()
                min_max["node_labels"][i] = (min(min_max["node_labels"][i][0], min_val), max(min_max["node_labels"][i][1], max_val))

        # Edge attributes
        if min_max["edge_attr"] is None:
            min_max["edge_attr"] = [(torch.min(data.edge_attr[:, i]).item(), torch.max(data.edge_attr[:, i]).item()) for i in range(data.edge_attr.shape[1])]
        elif data.edge_attr.size(0) > 0:
            for i in range(data.edge_attr.shape[1]):
                min_val, max_val = torch.min(data.edge_attr[:, i]).item(), torch.max(data.edge_attr[:, i]).item()
                min_max["edge_attr"][i] = (min(min_max["edge_attr"][i][0], min_val), max(min_max["edge_attr"][i][1], max_val))
        
        # Scalar labels
        min_max["y"][0] = min(min_max["y"][0], data.y.item() / 1000)
        min_max["y"][1] = max(min_max["y"][1], data.y.item() / 1000)
        min_max["y_cumulative"][0] = min(min_max["y_cumulative"][0], data.y_cummulative.item() / 1000)
        min_max["y_cumulative"][1] = max(min_max["y_cumulative"][1], data.y_cummulative.item() / 1000)

    return min_max

def compute_distributions(processed_path, global_min_max, num_bins=50):
    """Second pass: Compute histograms with consistent bin edges."""
    # Initialize histograms
    x_hist = [np.zeros(num_bins) for _ in global_min_max["x"]]
    node_labels_hist = [np.zeros(num_bins) for _ in global_min_max["node_labels"]]
    edge_attr_hist = [np.zeros(num_bins) for _ in global_min_max["edge_attr"]]
    edge_label_hist = np.zeros(num_bins)
    y_hist = np.zeros(num_bins)
    y_cumulative_hist = np.zeros(num_bins)

    # Define bin edges based on min and max values
    bin_x = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["x"]]
    bin_node_labels = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["node_labels"]]
    bin_edge_attr = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["edge_attr"]]
    bin_edge_labels = np.linspace(0, 1, num_bins + 1)
    bin_y = np.linspace(global_min_max["y"][0], global_min_max["y"][1], num_bins + 1)
    bin_y_cumulative = np.linspace(global_min_max["y_cumulative"][0], global_min_max["y_cumulative"][1], num_bins + 1)

    # Loop through scenario data
    for data in tqdm(load_data_from_scenario_folders(processed_path), desc="Computing Histograms"):
        for i in range(data.x.shape[1]):
            hist, _ = np.histogram(data.x[:, i].numpy(), bins=bin_x[i])
            x_hist[i] += hist

        for i in range(data.node_labels.shape[1]):
            hist, _ = np.histogram(data.node_labels[:, i].numpy(), bins=bin_node_labels[i])
            node_labels_hist[i] += hist

        for i in range(data.edge_attr.shape[1]):
            hist, _ = np.histogram(data.edge_attr[:, i].numpy(), bins=bin_edge_attr[i])
            edge_attr_hist[i] += hist
        

        edge_label_hist += np.histogram(data.edge_labels.numpy(), bins=bin_edge_labels)[0]
        y_hist += np.histogram([data.y / 1000], bins=bin_y)[0]
        y_cumulative_hist += np.histogram([data.y_cummulative / 1000], bins=bin_y_cumulative)[0]

    return x_hist, bin_x, node_labels_hist, bin_node_labels, edge_attr_hist, bin_edge_attr, y_hist, bin_y, y_cumulative_hist, bin_y_cumulative, edge_label_hist, bin_edge_labels

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
        'Edge_Label': {
            0: 'Status'
        },
        'Edge_Feature': {
            0: 'Edge_Real',
            1: 'Edge_Imag'
        },
        'Graph_label': {
            0: 'total_load_shed_step'
        },
        'Cumulative_Graph_Label': {
            0: 'total_load_shed_cumulative'
        }
    }
    for i, (h, b) in enumerate(zip(hist, bin_edges)):
        plt.bar(b[:-1], h, width=np.diff(b), align='edge', alpha=0.7)   
        plt.xlabel(lettering[title][i])
        plt.ylabel('Frequency')
        plt.title(lettering[title][i])
        plt.savefig(os.path.join(save_path, f'{NAME}_{title.lower()}_{i+1}.png'))
        plt.clf()


# Main Execution
if __name__ == "__main__":
    processed_path = "processed/"
    save_dir = "feat_dists/"
    NAME = 'edge_label_test'
    os.makedirs(save_dir, exist_ok=True)

    # First pass: Get global min and max values
    global_min_max = get_global_min_max(processed_path)

    # Second pass: Compute distributions
    results = compute_distributions(processed_path, global_min_max)

    # Plot and save histograms
    plot_histograms(results[0], results[1], "Feature", save_dir, NAME)
    plot_histograms(results[2], results[3], "Node_Label", save_dir, NAME)
    plot_histograms(results[4], results[5], "Edge_Feature", save_dir, NAME)

    # Plot scalar labels
    plt.bar(results[11][:-1], results[10], width=np.diff(results[11]), align='edge', alpha=0.7)
    plt.xlabel('Status')
    plt.ylabel('Frequency')
    plt.title('Edge Labels')
    plt.savefig(os.path.join(save_dir, f'{NAME}_edge_labels_distribution.png'))
    plt.clf()

    plt.bar(results[7][:-1], results[6], width=np.diff(results[7]), align='edge', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('y Distribution')
    plt.savefig(os.path.join(save_dir, f'{NAME}_y_distribution.png'))
    plt.clf()

    plt.bar(results[9][:-1], results[8], width=np.diff(results[9]), align='edge', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('y_cumulative Distribution')
    plt.savefig(os.path.join(save_dir, f'{NAME}_y_cumulative_distribution.png'))
    plt.clf()
