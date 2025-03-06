import os
import torch
import shutil
import numpy as np
from tqdm import tqdm


# Thresholds for checks (customize these values)
thresholds = {
    "x": {"min": [-300, -220, -550, -550], "max": [1500, 550, 550, 550]},  # First 4 node features
    "node_labels": {"min": -550, "max": 550},  # Node labels
    "edge_attr": {"min": [-200, -3500], "max": [350, 2200]},  # Edge attributes
}


def check_for_invalid_values(file_path):
    """
    Checks a .pt file for NaNs, positive/negative infinities, and threshold violations.
    Returns details about where invalid values were found.
    """
    data = torch.load(file_path)
    invalid_details = []

    # Check node features (x)
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        invalid_details.append(
            f"x: NaNs: {torch.isnan(data.x).nonzero(as_tuple=False).tolist()}, "
            f"Infs: {torch.isinf(data.x).nonzero(as_tuple=False).tolist()}"
        )
    if data.x.shape[1] >= 4:  # Check first 4 node features for threshold violations
        for i in range(4):
            if (data.x[:, i] < thresholds["x"]["min"][i]).any() or (data.x[:, i] > thresholds["x"]["max"][i]).any():
                invalid_indices = torch.nonzero(
                    (data.x[:, i] < thresholds["x"]["min"][i]) | (data.x[:, i] > thresholds["x"]["max"][i]),
                    as_tuple=False,
                )
                invalid_details.append(
                    f"x[:, {i}]: Threshold violation at indices {invalid_indices.tolist()}"
                )

    # Check node labels
    if torch.isnan(data.node_labels).any() or torch.isinf(data.node_labels).any():
        invalid_details.append(
            f"node_labels: NaNs: {torch.isnan(data.node_labels).nonzero(as_tuple=False).tolist()}, "
            f"Infs: {torch.isinf(data.node_labels).nonzero(as_tuple=False).tolist()}"
        )
    if (data.node_labels < thresholds["node_labels"]["min"]).any() or (
        data.node_labels > thresholds["node_labels"]["max"]
    ).any():
        invalid_indices = torch.nonzero(
            (data.node_labels < thresholds["node_labels"]["min"])
            | (data.node_labels > thresholds["node_labels"]["max"]),
            as_tuple=False,
        )
        invalid_details.append(
            f"node_labels: Threshold violation at indices {invalid_indices.tolist()}"
        )

    # Check edge attributes
    if data.edge_attr.size(0) > 0:
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            invalid_details.append(
                f"edge_attr: NaNs: {torch.isnan(data.edge_attr).nonzero(as_tuple=False).tolist()}, "
                f"Infs: {torch.isinf(data.edge_attr).nonzero(as_tuple=False).tolist()}"
            )
        if data.edge_attr.shape[1] >= 2:
            for i in range(2):
                if (data.edge_attr[:,i] < thresholds["edge_attr"]["min"][i]).any() or (
                    data.edge_attr[:,i] > thresholds["edge_attr"]["max"][i]).any():
                    invalid_indices = torch.nonzero(
                        (data.edge_attr[:,i] < thresholds["edge_attr"]["min"][i])
                        | (data.edge_attr[:,i] > thresholds["edge_attr"]["max"][i]),
                        as_tuple=False,
                    )
                    invalid_details.append(
                        f"edge_attr: Threshold violation at indices {invalid_indices.tolist()}"
                    )

    # Check graph labels (y)
    if torch.isnan(data.y).any() or torch.isinf(data.y).any():
        invalid_details.append("y contains invalid values")

    # Check cumulative labels (y_cumulative)
    if torch.isnan(data.y_cummulative).any() or torch.isinf(data.y_cummulative).any():
        invalid_details.append("y_cummulative contains invalid values")

    return invalid_details


def remove_invalid_scenarios(processed_path, dump_path):
    """
    Scans for NaNs, positive/negative infinities, and threshold violations, 
    and removes entire scenario folders if any file contains invalid values.
    """
    os.makedirs(dump_path, exist_ok=True)
    scenarios_to_remove = set()

    for root, dirs, files in os.walk(processed_path):
        scenario_files = [f for f in files if f.endswith(".pt") and "pre" not in f]

        for file in tqdm(scenario_files, desc=f"Checking files in {root}"):
            file_path = os.path.join(root, file)
            invalid_details = check_for_invalid_values(file_path)

            if invalid_details:
                scenario_folder = os.path.relpath(root, processed_path)
                scenarios_to_remove.add(scenario_folder)
                print(f"Invalid values detected in {file_path}: {', '.join(invalid_details)}")

    # Move folders with invalid values to the dump folder
    for scenario in scenarios_to_remove:
        scenario_path = os.path.join(processed_path, scenario)
        dump_scenario_path = os.path.join(dump_path, scenario)
        print(f"Moving scenario folder with invalid values: {scenario_path} -> {dump_scenario_path}")
        shutil.move(scenario_path, dump_scenario_path)


if __name__ == "__main__":
    processed_path = "processed/"
    dump_path = "dump/"

    print("Scanning for NaNs, Infs, and threshold violations in dataset...")
    remove_invalid_scenarios(processed_path, dump_path)
    print("Invalid value removal process completed.")
