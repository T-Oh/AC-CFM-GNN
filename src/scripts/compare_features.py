import torch
import os
import numpy as np
import matplotlib.pyplot as plt

#Script used to compare the features of two datasets

# Variable to differentiate datasets
NAME = 'dataset1'  # Change this for different datasets

FOLDER_ANGF_CE_Y = 'ANGF_CE_Y/'
FOLDER_ZHU = 'processed/'

def create_edge_dict(edges, edge_features):
    """Create a dictionary with edge tuple as the key and the feature as the value."""
    edge_dict = {}
    for i in range(edges.shape[1]):
        edge = tuple(edges[:, i])
        edge_dict[edge] = edge_features[i]
    return edge_dict

# Initialize histogram variables
first_Y, first_P, first_Q = True, True, True
first_bus_type = [True] * 4

for file in os.listdir(FOLDER_ANGF_CE_Y):
    if file.startswith('data') and os.path.exists(FOLDER_ZHU+file):
        print(file)
        # Load both files
        zhu = torch.load(FOLDER_ZHU+file)
        angf = torch.load(FOLDER_ANGF_CE_Y+file)
        
        # Get sorted adjacency matrices
        adj_zhu = np.sort(zhu.edge_index, axis=0)
        adj_angf = np.sort(angf.edge_index, axis=0)
        
        # Recalculate zhu Y to real values
        Y_old_zhu = torch.sqrt(zhu.edge_attr[:,0]**2 + zhu.edge_attr[:,1]**2)
        
        # Create Y dicts for easier comparison
        edge_dict_zhu = create_edge_dict(adj_zhu, Y_old_zhu)
        edge_dict_angf = create_edge_dict(adj_angf, angf.edge_attr)
        
        # Get Power injection
        P_zhu = zhu.x[:,0]
        Q_zhu = zhu.x[:,1]
        P_angf = angf.x[:,8] - angf.x[:,0]  # Pg - Pd
        Q_angf = angf.x[:,9] - angf.x[:,1]  # Qg - Qd
        
        # Get bus types
        bus_types_zhu = zhu.x[:,4:]
        bus_type_angf = angf.x[:,4:8]

        # Compare edges and their features
        for edge, Y_zhu in edge_dict_zhu.items():
            if edge in edge_dict_angf:
                Y_angf = edge_dict_angf[edge]
                Ydiff = Y_zhu - Y_angf
                rel_Ydiff = Ydiff / Y_zhu
                
                if first_Y:
                    rel_Ydiff_hist, rel_Ydiff_bins = np.histogram(rel_Ydiff, bins=100, range=[0, 1])
                    first_Y = False
                else:
                    rel_Ydiff_hist_temp, _ = np.histogram(rel_Ydiff, bins=rel_Ydiff_bins)
                    rel_Ydiff_hist += rel_Ydiff_hist_temp

        # Compare node features
        P_diff = P_zhu - P_angf
        Q_diff = Q_zhu - Q_angf
        bus_type_diff = bus_types_zhu - bus_type_angf
        rel_P_diff = P_diff / P_zhu
        rel_Q_diff = Q_diff / Q_zhu

        # Create histograms for P and Q differences
        if first_P:
            rel_Pdiff_hist, rel_Pdiff_bins = np.histogram(rel_P_diff.numpy(), bins=100, range=[-1, 1])
            first_P = False
        else:
            rel_Pdiff_hist_temp, _ = np.histogram(rel_P_diff.numpy(), bins=rel_Pdiff_bins)
            rel_Pdiff_hist += rel_Pdiff_hist_temp
        
        if first_Q:
            rel_Qdiff_hist, rel_Qdiff_bins = np.histogram(rel_Q_diff.numpy(), bins=100, range=[-1, 1])
            first_Q = False
        else:
            rel_Qdiff_hist_temp, _ = np.histogram(rel_Q_diff.numpy(), bins=rel_Qdiff_bins)
            rel_Qdiff_hist += rel_Qdiff_hist_temp

        # Create histograms for bus type differences
        for i in range(bus_types_zhu.shape[1]):
            bus_type_diff_i = bus_type_diff[:, i].numpy()
            if first_bus_type[i]:
                bus_type_hist, bus_type_bins = np.histogram(bus_type_diff_i, bins=100, range=[-1, 1])
                first_bus_type[i] = False
            else:
                bus_type_hist_temp, _ = np.histogram(bus_type_diff_i, bins=bus_type_bins)
                bus_type_hist += bus_type_hist_temp

# Plot histogram of relative Y differences
plt.figure(figsize=(8, 6))
plt.bar(rel_Ydiff_bins[:-1], rel_Ydiff_hist, width=rel_Ydiff_bins[1] - rel_Ydiff_bins[0])
plt.xlabel('Relative Difference in Y')
plt.ylabel('Number of Edges')
plt.title('Relative Y Differences Between Datasets')
plt.savefig(f'rel_Ydiff_{NAME}.png')
plt.show()

# Plot histogram of relative P differences
plt.figure(figsize=(8, 6))
plt.bar(rel_Pdiff_bins[:-1], rel_Pdiff_hist, width=rel_Pdiff_bins[1] - rel_Pdiff_bins[0])
plt.xlabel('Relative Difference in P')
plt.ylabel('Number of Buses')
plt.title('Relative P Differences Between Datasets')
plt.savefig(f'rel_P_diff_{NAME}.png')
plt.show()

# Plot histogram of relative Q differences
plt.figure(figsize=(8, 6))
plt.bar(rel_Qdiff_bins[:-1], rel_Qdiff_hist, width=rel_Qdiff_bins[1] - rel_Qdiff_bins[0])
plt.xlabel('Relative Difference in Q')
plt.ylabel('Number of Buses')
plt.title('Relative Q Differences Between Datasets')
plt.savefig(f'rel_Q_diff_{NAME}.png')
plt.show()

# Plot histograms for bus type differences
bus_type_labels = ['Bus Type 1', 'Bus Type 2', 'Bus Type 3', 'Bus Type 4']
for i in range(len(bus_type_labels)):
    plt.figure(figsize=(8, 6))
    plt.bar(bus_type_bins[:-1], bus_type_hist, width=bus_type_bins[1] - bus_type_bins[0])
    plt.xlabel(f'Difference in {bus_type_labels[i]}')
    plt.ylabel('Number of Buses')
    plt.title(f'{bus_type_labels[i]} Differences Between Datasets')
    plt.savefig(f'bus_type_{i}_diff_{NAME}.png')
    plt.show()
