import os
import torch
import numpy as np
from tqdm import tqdm
import re

def natural_sort_key(filename):
    """Key function for natural sorting, splitting text and numbers."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def compute_min_max(processed_path):
    """Computes the min and max values for each feature and label."""
    min_values = None
    max_values = None

    for root, dirs, files in os.walk(processed_path):
        # Natural sorting of files
        scenario_files = sorted(
            [f for f in files if f.endswith(".pt") and "pre" not in f],
            key=natural_sort_key
        )
        for i, file in enumerate(scenario_files):
            file_path = os.path.join(root, file)
            data = torch.load(file_path)

            if min_values is None:
                min_values = {
                    "x": np.min(data.x.numpy(), axis=0),
                    "node_labels": np.min(data.node_labels.numpy(), axis=0),
                    "y": data.y,
                    "y_cumulative": data.y_cummulative
                }
                max_values = {
                    "x": np.max(data.x.numpy(), axis=0),
                    "node_labels": np.max(data.node_labels.numpy(), axis=0),
                    "y": data.y,
                    "y_cumulative": data.y_cummulative
                }
            else:
                min_values["x"] = np.minimum(min_values["x"], np.min(data.x.numpy(), axis=0))
                max_values["x"] = np.maximum(max_values["x"], np.max(data.x.numpy(), axis=0))

                min_values["node_labels"] = np.minimum(min_values["node_labels"], np.min(data.node_labels.numpy(), axis=0))
                max_values["node_labels"] = np.maximum(max_values["node_labels"], np.max(data.node_labels.numpy(), axis=0))

                min_values["y"] = min(min_values["y"], data.y)
                max_values["y"] = max(max_values["y"], data.y)

                if i == len(scenario_files) - 1:
                    min_values["y_cumulative"] = min(min_values["y_cumulative"], data.y_cummulative)
                    max_values["y_cumulative"] = max(max_values["y_cumulative"], data.y_cummulative)

    return min_values, max_values

def normalize_data(data, min_values, max_values, log_normalize, num_features):
    """Normalizes data using min-max or log normalization."""
    # Normalize node features
    if log_normalize:
        data.x = torch.log1p(data.x)
    else:
        data.x = (data.x - torch.tensor(min_values["x"])) / (torch.tensor(max_values["x"]) - torch.tensor(min_values["x"]))
    
    # Normalize node labels
    if log_normalize:
        data.node_labels = torch.log1p(data.node_labels)
    else:
        data.node_labels = (data.node_labels - torch.tensor(min_values["node_labels"])) / (
            torch.tensor(max_values["node_labels"]) - torch.tensor(min_values["node_labels"])
        )

    # Normalize graph labels
    #data.y = (data.y - min_values["y"]) / (max_values["y"] - min_values["y"])
    data.y = torch.log1p(data.y)/torch.log1p(torch.tensor(max_values["y"]))
    data.y_cummulative = (data.y_cummulative - min_values["y_cumulative"]) / (
        max_values["y_cumulative"] - min_values["y_cumulative"]
    )

    return data

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
        for i, file in enumerate(scenario_files):
            file_path = os.path.join(root, file)
            data = torch.load(file_path)

            # Normalize data
            normalized_data = normalize_data(data, min_values, max_values, log_normalize, data.x.shape[1])

            # Save normalized data
            save_file_path = os.path.join(save_path, file)
            torch.save(normalized_data, save_file_path)

if __name__ == "__main__":
    processed_path = "processed/"
    normalized_path = "normalized/"
    log_normalize = False  # Set to True for log normalization

    # Step 1: Compute min and max values
    print("Computing min and max values...")
    min_values, max_values = compute_min_max(processed_path)

    # Step 2: Normalize and save data
    print("Normalizing and saving data...")
    process_and_save(processed_path, normalized_path, min_values, max_values, log_normalize)

    print("Normalization complete.")
