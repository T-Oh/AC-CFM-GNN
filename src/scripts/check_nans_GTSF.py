import os
import torch
import shutil
import numpy as np
from tqdm import tqdm

def check_for_nans(file_path):
    """Checks a .pt file for NaNs and returns details about where NaNs were found."""
    data = torch.load(file_path)
    nan_details = []

    if torch.isnan(data.x).any():
        nan_details.append(f"x: {torch.isnan(data.x).nonzero(as_tuple=False).tolist()}")
    if torch.isnan(data.node_labels).any():
        nan_details.append(f"node_labels: {torch.isnan(data.node_labels).nonzero(as_tuple=False).tolist()}")
    if data.edge_attr.size(0) > 0 and torch.isnan(data.edge_attr).any():
        nan_details.append(f"edge_attr: {torch.isnan(data.edge_attr).nonzero(as_tuple=False).tolist()}")
    if torch.isnan(data.y).any():
        nan_details.append("y")
    if torch.isnan(data.y_cummulative).any():
        nan_details.append("y_cummulative")
    
    return nan_details

def remove_nan_scenarios(processed_path, dump_path):
    """Scans for NaNs and removes entire scenario folders if any file contains NaNs."""
    os.makedirs(dump_path, exist_ok=True)
    scenarios_to_remove = set()

    for root, dirs, files in os.walk(processed_path):
        scenario_files = [f for f in files if f.endswith(".pt") and "pre" not in f]

        for file in tqdm(scenario_files, desc=f"Checking files in {root}"):
            file_path = os.path.join(root, file)
            nan_details = check_for_nans(file_path)

            if nan_details:
                scenario_folder = os.path.relpath(root, processed_path)
                scenarios_to_remove.add(scenario_folder)
                print(f"NaN detected in {file_path}: {', '.join(nan_details)}")

    # Move folders with NaNs to the dump folder
    for scenario in scenarios_to_remove:
        scenario_path = os.path.join(processed_path, scenario)
        dump_scenario_path = os.path.join(dump_path, scenario)
        print(f"Moving scenario folder with NaNs: {scenario_path} -> {dump_scenario_path}")
        shutil.move(scenario_path, dump_scenario_path)

if __name__ == "__main__":
    processed_path = "processed/"
    dump_path = "dump/"

    print("Scanning for NaNs in dataset...")
    remove_nan_scenarios(processed_path, dump_path)
    print("NaN removal process completed.")
