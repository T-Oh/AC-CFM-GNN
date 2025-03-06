import os
import torch
import numpy as np
from tqdm import tqdm
import re
import pickle

def natural_sort_key(filename):
    """Key function for natural sorting, splitting text and numbers."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def save_min_max(file_path, min_values, max_values):
    """Saves the min and max values to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump({'min_values': min_values, 'max_values': max_values}, f)

def load_min_max(file_path):
    """Loads the min and max values from a file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['min_values'], data['max_values']

def process_and_save(processed_path, normalized_path, min_values, max_values, log_normalize):
    """Processes and normalizes data incrementally, saving results."""
    os.makedirs(normalized_path, exist_ok=True)

    for root, dirs, files in os.walk(processed_path):
        # Ensure folder structure is mirrored
        rel_path = os.path.relpath(root, processed_path)
        save_path = os.path.join(normalized_path, rel_path)
        os.makedirs(save_path, exist_ok=True)

        # Natural sorting of files
        scenario_files = sorted(
            [f for f in files if f.endswith(".pt") and "pre" not in f],
            key=natural_sort_key
        )
        for file in scenario_files:
            file_path = os.path.join(root, file)
            data = torch.load(file_path)

            # Normalize data
            normalized_data = normalize_data(data, min_values, max_values, log_normalize, data.x.shape[1])

            # Save normalized data
            save_file_path = os.path.join(save_path, file)
            torch.save(normalized_data, save_file_path)

def compute_min_max(processed_path):
    """Computes the min and max values for each feature, label, and edge feature."""
    min_values = None
    max_values = None

    for root, dirs, files in os.walk(processed_path):
        # Natural sorting of files
        scenario_files = sorted(
            [f for f in files if f.endswith(".pt") and "pre" not in f],
            key=natural_sort_key
        )
        for file in scenario_files:
            file_path = os.path.join(root, file)
            data = torch.load(file_path)

            if min_values is None:
                min_values = {
                    "x": np.min(data.x.numpy(), axis=0),
                    "node_labels": np.min(data.node_labels.numpy(), axis=0),
                    "edge_attr": np.min(data.edge_attr.numpy(), axis=0) if data.edge_attr.size(0) > 0 else None,
                    "y": data.y,
                    "y_cummulative": data.y_cummulative
                }
                max_values = {
                    "x": np.max(data.x.numpy(), axis=0),
                    "node_labels": np.max(data.node_labels.numpy(), axis=0),
                    "edge_attr": np.max(data.edge_attr.numpy(), axis=0) if data.edge_attr.size(0) > 0 else None,
                    "y": data.y,
                    "y_cummulative": data.y_cummulative
                }
            else:
                min_values["x"] = np.minimum(min_values["x"], np.min(data.x.numpy(), axis=0))
                max_values["x"] = np.maximum(max_values["x"], np.max(data.x.numpy(), axis=0))

                min_values["node_labels"] = np.minimum(min_values["node_labels"], np.min(data.node_labels.numpy(), axis=0))
                max_values["node_labels"] = np.maximum(max_values["node_labels"], np.max(data.node_labels.numpy(), axis=0))

                if data.edge_attr.size(0) > 0:
                    min_values["edge_attr"] = np.minimum(min_values["edge_attr"], np.min(data.edge_attr.numpy(), axis=0))
                    max_values["edge_attr"] = np.maximum(max_values["edge_attr"], np.max(data.edge_attr.numpy(), axis=0))

                min_values["y"] = min(min_values["y"], data.y)
                max_values["y"] = max(max_values["y"], data.y)

                min_values["y_cummulative"] = min(min_values["y_cummulative"], data.y_cummulative)
                max_values["y_cummulative"] = max(max_values["y_cummulative"], data.y_cummulative)
    
    return min_values, max_values

def normalize_data(data, min_values, max_values, log_normalize, num_features):
    """Normalizes data using min-max or log normalization."""
    # Normalize only the first 4 node features
    if log_normalize:
        data.x[:, :4] = torch.log1p(data.x[:, :4]-torch.tensor(min_values["x"][:4])) / torch.log1p(torch.tensor(max_values["x"][:4]) - torch.tensor(min_values["x"][:4])) 
    else:
        data.x[:, :4] = (data.x[:, :4] - torch.tensor(min_values["x"][:4])) / (
            torch.tensor(max_values["x"][:4]) - torch.tensor(min_values["x"][:4])
        )
    
    # Normalize node labels
    if log_normalize:
        data.node_labels = torch.log1p(data.node_labels - torch.tensor(min_values["node_labels"])) / torch.log1p(torch.tensor(max_values["node_labels"]) - torch.tensor(min_values["node_labels"]))
    else:
        data.node_labels = (data.node_labels - torch.tensor(min_values["node_labels"])) / (
            torch.tensor(max_values["node_labels"]) - torch.tensor(min_values["node_labels"])
        )
    log_normalize = False
    # Normalize edge features if not empty
    if data.edge_attr.size(0) > 0:
        if log_normalize:
            data.edge_attr = torch.log1p(data.edge_attr)
        else:
            data.edge_attr = (data.edge_attr - torch.tensor(min_values["edge_attr"])) / (
                torch.tensor(max_values["edge_attr"]) - torch.tensor(min_values["edge_attr"])
            )
    
    # Normalize graph labels
    data.y = torch.log1p(data.y) / torch.log1p(torch.tensor(max_values["y"]))
    data.y_cummulative = (data.y_cummulative - min_values["y_cummulative"]) / (
        max_values["y_cummulative"] - min_values["y_cummulative"]
    )

    return data

if __name__ == "__main__":
    processed_path = "processed/"
    normalized_path = "normalized/"
    min_max_file = "min_max_ASnoCl.pkl"
    recalculate_min_max = True  # Set to True to recalculate min/max values
    log_normalize = False  # Set to True for log normalization

    # Step 1: Compute or load min and max values
    if recalculate_min_max or not os.path.exists(min_max_file):
        print("Computing min and max values...")
        min_values, max_values = compute_min_max(processed_path)
        save_min_max(min_max_file, min_values, max_values)
    else:
        print("Loading min and max values...")
        min_values, max_values = load_min_max(min_max_file)
    # Step 2: Normalize and save data
    print("Normalizing and saving data...")
    process_and_save(processed_path, normalized_path, min_values, max_values, log_normalize)

    print("Normalization complete.")
