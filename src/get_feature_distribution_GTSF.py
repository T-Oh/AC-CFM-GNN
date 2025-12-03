import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import pandas as pd
import re

import re

def load_data_from_scenario_folders(processed_path, sequence_length=None):
    """Generator to load data files from scenario folders, marking the final step.
    Only processes scenarios whose ID does NOT start with 1.
    """
    pattern = re.compile(r"^scenario_([0-9]+)$")

    for root, dirs, files in os.walk(processed_path):
        # Check if this folder is a scenario folder
        folder_name = os.path.basename(root)
        match = pattern.match(folder_name)
        if match:
            scenario_id = match.group(1)
            if scenario_id.startswith("1"):
                # Skip scenario IDs that start with 1
                continue

        if not files or "data_static.py" in files:
            continue

        pt_files = [f for f in files if f.endswith(".pt") and "pre" not in f]
        if not pt_files:
            continue

        # Sort files naturally by step index
        try:
            pt_files_sorted = sorted(
                pt_files,
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
        except Exception:
            pt_files_sorted = sorted(pt_files)

        # Keep only the last N steps if requested
        if sequence_length is not None and sequence_length > 0:
            pt_files_sorted = pt_files_sorted[-sequence_length:]

        last_file = pt_files_sorted[-1]  # final step of this scenario

        for file in pt_files_sorted:
            file_path = os.path.join(root, file)
            try:
                data = torch.load(file_path)
                yield data, (file == last_file)
            except Exception as e:
                print(f"[CORRUPTED] Could not load {file_path}: {e}")
                continue



def get_global_min_max(processed_path, sequence_length):
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
    for data, is_final in tqdm(load_data_from_scenario_folders(processed_path, sequence_length),
                            desc="Scanning Min/Max"):
        # === Features ===
        if min_max["x"] is None:
            min_max["x"] = [(torch.min(data.x[:, i]).item(), torch.max(data.x[:, i]).item()) 
                            for i in range(data.x.shape[1])]
        else:
            for i in range(data.x.shape[1]):
                min_val, max_val = torch.min(data.x[:, i]).item(), torch.max(data.x[:, i]).item()
                min_max["x"][i] = (min(min_max["x"][i][0], min_val),
                                max(min_max["x"][i][1], max_val))

        # === Edge attributes ===
        if min_max["edge_attr"] is None:
            min_max["edge_attr"] = [(torch.min(data.edge_attr[:, i]).item(), torch.max(data.edge_attr[:, i]).item()) 
                                    for i in range(data.edge_attr.shape[1])]
        elif data.edge_attr.size(0) > 0:
            for i in range(data.edge_attr.shape[1]):
                min_val, max_val = torch.min(data.edge_attr[:, i]).item(), torch.max(data.edge_attr[:, i]).item()
                min_max["edge_attr"][i] = (min(min_max["edge_attr"][i][0], min_val),
                                        max(min_max["edge_attr"][i][1], max_val))

        # === Node labels only from final step ===
        if is_final:
            if min_max["node_labels"] is None:
                min_max["node_labels"] = [(torch.min(data.node_labels[:, i]).item(), torch.max(data.node_labels[:, i]).item()) 
                                        for i in range(data.node_labels.shape[1])]
            else:
                for i in range(data.node_labels.shape[1]):
                    min_val, max_val = torch.min(data.node_labels[:, i]).item(), torch.max(data.node_labels[:, i]).item()
                    min_max["node_labels"][i] = (min(min_max["node_labels"][i][0], min_val),
                                                max(min_max["node_labels"][i][1], max_val))

    return min_max



def compute_distributions(processed_path, global_min_max, num_bins=50, sequence_length=None):
    """Compute histograms and feature statistics in afor data in tqdm(load_data_from_scenario_folders(processed_path), desc="Computing Histograms & Stats"): single data pass."""
    # Initialize histograms
    x_hist = [np.zeros(num_bins) for _ in global_min_max["x"]]
    node_labels_hist = [np.zeros(num_bins) for _ in global_min_max["node_labels"]]
    edge_attr_hist = [np.zeros(num_bins) for _ in global_min_max["edge_attr"]]
    edge_label_hist = np.zeros(num_bins)
    y_hist = np.zeros(num_bins)
    y_cumulative_hist = np.zeros(num_bins)

    # Initialize stats accumulators
    x_stats = [{"values": []} for _ in global_min_max["x"]]
    node_stats = [{"values": []} for _ in global_min_max["node_labels"]]
    edge_stats = [{"values": []} for _ in global_min_max["edge_attr"]]
    y_values = []
    ycum_values = []

    # Define bin edges based on min/max
    bin_x = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["x"]]
    bin_node_labels = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["node_labels"]]
    bin_edge_attr = [np.linspace(mn, mx, num_bins + 1) for mn, mx in global_min_max["edge_attr"]]
    bin_edge_labels = np.linspace(0, 1, num_bins + 1)
    bin_y = np.linspace(global_min_max["y"][0], global_min_max["y"][1], num_bins + 1)
    bin_y_cumulative = np.linspace(global_min_max["y_cumulative"][0], global_min_max["y_cumulative"][1], num_bins + 1)

    # Loop through scenario data
    for data, is_final in tqdm(load_data_from_scenario_folders(processed_path, sequence_length),
                            desc="Computing Histograms & Stats"):
        # === Features (all last N steps) ===
        for i in range(data.x.shape[1]):
            arr = data.x[:, i].cpu().numpy()
            hist, _ = np.histogram(arr, bins=bin_x[i])
            x_hist[i] += hist
            x_stats[i]["values"].append(arr)

        # === Edge attributes (all last N steps) ===
        for i in range(data.edge_attr.shape[1]):
            arr = data.edge_attr[:, i].cpu().numpy()
            hist, _ = np.histogram(arr, bins=bin_edge_attr[i])
            edge_attr_hist[i] += hist
            edge_stats[i]["values"].append(arr)

        # === Node labels (only final step) ===
        if is_final:
            for i in range(data.node_labels.shape[1]):
                arr = data.node_labels[:, i].cpu().numpy()
                hist, _ = np.histogram(arr, bins=bin_node_labels[i])
                node_labels_hist[i] += hist
                node_stats[i]["values"].append(arr)


    # === Compute statistics ===
    def summarize(values_list):
        arr = np.concatenate(values_list)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "skew": float(skew(arr, nan_policy="omit")),
            "kurtosis": float(kurtosis(arr, nan_policy="omit"))
        }

    x_summary = [summarize(stat["values"]) for stat in x_stats]
    node_summary = [summarize(stat["values"]) for stat in node_stats]
    edge_summary = [summarize(stat["values"]) for stat in edge_stats]
    #y_summary = [summarize([np.array(y_values)])]
    #ycum_summary = [summarize([np.array(ycum_values)])]

    # === Combine all statistics ===
    all_stats = {
        "x": x_summary,
        "node_labels": node_summary,
        "edge_attr": edge_summary,
        #"y": y_summary,
        #"y_cumulative": ycum_summary
    }

    return (
        x_hist, bin_x,
        node_labels_hist, bin_node_labels,
        edge_attr_hist, bin_edge_attr,
        y_hist, bin_y,
        y_cumulative_hist, bin_y_cumulative,
        edge_label_hist, bin_edge_labels,
        all_stats
    )

def save_statistics_to_csv(stats, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    for key, stat_list in stats.items():
        df = pd.DataFrame(stat_list)
        df.to_csv(os.path.join(save_path, f"{name}_{key}_statistics.csv"), index=False)

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
    NAME = 'PU_test_normalized' 
    sequence_length = 5
    os.makedirs(save_dir, exist_ok=True)

    # First pass: Get global min and max values
    global_min_max = get_global_min_max(processed_path, sequence_length)

    # Second pass: Compute distributions
    results = compute_distributions(processed_path, global_min_max, sequence_length)

    # Unpack results (last element = all_stats)
    (
        x_hist, bin_x,
        node_labels_hist, bin_node_labels,
        edge_attr_hist, bin_edge_attr,
        y_hist, bin_y,
        y_cumulative_hist, bin_y_cumulative,
        edge_label_hist, bin_edge_labels,
        feature_stats
    ) = results

    # Save statistics
    save_statistics_to_csv(feature_stats, save_dir, NAME)


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
