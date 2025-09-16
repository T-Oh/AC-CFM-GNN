# %%
import torch
import os
import numpy as np
import h5py
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix
from torch_scatter import scatter_add
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error
from torch_sparse import coalesce





# %%
# Verbosity flag 
VERBOSE = True
PATH = '/home/tohlinger/Documents/hi-accf-ml/'
output = torch.load(PATH + 'results/output_final.pt')
labels = torch.load(PATH + 'results/labels_final.pt')
static = torch.load(PATH+'processed/scenario_1/data_1_43.pt') #load static for original edge attributes and indices
#static_edge_attr = torch.load('static_edge_attr_cluster.pt') #load static edge attributes
with open('/home/tohlinger/Documents/hi-accf-ml/min_max_ASnoCl.pkl', 'rb') as f:
    min_max = pkl.load(f)   #load min_max values for denormalization
if VERBOSE:
    print('RAW (NORMALIZED):')
    print('OUTPUT')
    print(output[0][:10])
    print('LABELS')
    print(labels[0][:10])







def collect_predictions_by_degree(S_all, S_true_all, edge_labels, static):
    """
    Groups predictions and true values by node degree.
    
    Parameters:
    - S_all: list of complex tensors (model predictions per instance)
    - S_true_all: list of complex tensors (ground truth per instance)
    - edge_index_list: list of edge_index tensors per instance
    
    Returns:
    - pred_by_deg_real, true_by_deg_real: dicts of np.arrays per degree
    - pred_by_deg_imag, true_by_deg_imag: dicts of np.arrays per degree
    """
    pred_by_deg_real = defaultdict(list)
    true_by_deg_real = defaultdict(list)
    pred_by_deg_imag = defaultdict(list)
    true_by_deg_imag = defaultdict(list)
    #ctive_mask = (edge_labels == 1)

    #updated_edge_attr = static.edge_attr[active_mask, :]
    updated_edge_index_list = []
    for i in range(edge_labels.shape[0]):
        active_mask = edge_labels[i] == 1  # shape: [N_EDGES]
        # Filter the base edge_index for this instance
        #filtered_edge_index = static.edge_index[:, active_mask]  # shape: [2, num_active_edges]
        #updated_edge_index_list.append(filtered_edge_index)
        updated_edge_index_list.append(static.edge_index)
    for S_pred, S_true, edge_index in zip(S_all, S_true_all, updated_edge_index_list):
        # Infer degrees from edge_index

        degrees = torch.bincount(edge_index[0], minlength=S_pred.shape[0])

        for node_idx, deg in enumerate(degrees.tolist()):
            pred_by_deg_real[deg].append(S_pred[node_idx].real.item())
            true_by_deg_real[deg].append(S_true[node_idx].real.item())
            pred_by_deg_imag[deg].append(S_pred[node_idx].imag.item())
            true_by_deg_imag[deg].append(S_true[node_idx].imag.item())

    # Convert lists to numpy arrays
    for d in pred_by_deg_real:
        pred_by_deg_real[d] = np.array(pred_by_deg_real[d])
        true_by_deg_real[d] = np.array(true_by_deg_real[d])
        pred_by_deg_imag[d] = np.array(pred_by_deg_imag[d])
        true_by_deg_imag[d] = np.array(true_by_deg_imag[d])

    return pred_by_deg_real, true_by_deg_real, pred_by_deg_imag, true_by_deg_imag

def compute_and_plot_metrics(pred_by_deg_real, true_by_deg_real,
                             pred_by_deg_imag, true_by_deg_imag, PATH):
    # Step 1: Compute R² and MSE per degree
    r2_by_deg_real = {}
    mse_by_deg_real = {}
    r2_by_deg_imag = {}
    mse_by_deg_imag = {}

    for deg in pred_by_deg_real:
        y_pred_real = pred_by_deg_real[deg]
        y_true_real = true_by_deg_real[deg]
        y_pred_imag = pred_by_deg_imag[deg]
        y_true_imag = true_by_deg_imag[deg]

        if len(y_true_real) > 1:
            r2_by_deg_real[deg] = r2_score(y_true_real, y_pred_real)
            mse_by_deg_real[deg] = mean_squared_error(y_true_real, y_pred_real)
            r2_by_deg_imag[deg] = r2_score(y_true_imag, y_pred_imag)
            mse_by_deg_imag[deg] = mean_squared_error(y_true_imag, y_pred_imag)
        else:
            r2_by_deg_real[deg] = np.nan
            mse_by_deg_real[deg] = np.nan
            r2_by_deg_imag[deg] = np.nan
            mse_by_deg_imag[deg] = np.nan

    # Step 2: Print results
    print("=== Real Part ===")
    for deg in sorted(r2_by_deg_real):
        print(f"Degree {deg}: R² = {r2_by_deg_real[deg]:.3f}, MSE = {mse_by_deg_real[deg]:.4e}")
    print("\n=== Imaginary Part ===")
    for deg in sorted(r2_by_deg_imag):
        print(f"Degree {deg}: R² = {r2_by_deg_imag[deg]:.3f}, MSE = {mse_by_deg_imag[deg]:.4e}")

    # Step 3: Plotting
    def plot_metrics(r2_by_deg, mse_by_deg, component='real'):
        degrees = sorted(r2_by_deg.keys())
        r2_vals = [r2_by_deg[d] for d in degrees]
        mse_vals = [mse_by_deg[d] for d in degrees]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Node Degree')
        ax1.set_ylabel(f'R² ({component})', color=color)
        ax1.plot(degrees, r2_vals, alpha=0.7, color=color, label='R²')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.set_ylim([-1.0, 1.0])

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(f'MSE ({component})', color=color)
        ax2.plot(degrees, mse_vals, 'o--', color=color, label='MSE')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')  # Log scale for MSE

        plt.title(f'R² and MSE by Node Degree ({component})')
        fig.tight_layout()
        fig.savefig(PATH+"metrics_by_degree.png")


    plot_metrics(r2_by_deg_real, mse_by_deg_real, component='real')
    plot_metrics(r2_by_deg_imag, mse_by_deg_imag, component='imag')

    return r2_by_deg_real, mse_by_deg_real, r2_by_deg_imag, mse_by_deg_imag



def plot_confusion_matrix_power_predictions(y_true_all, y_pred_all, title, path, n_bins=10):
    """
    Plots and saves two confusion matrices (real and imaginary parts) for binned power predictions.

    Parameters:
    - y_true_all: np.ndarray of true values (complex)
    - y_pred_all: np.ndarray of predicted values (complex)
    - title: string identifier for the model
    - path: directory path where the plot should be saved
    - n_bins: number of equal-width bins for true values (default 10)
    """
    def _plot_single_confusion(y_true_part, y_pred_part, part_label):
        # Create output directory
        confusion_dir = os.path.join(path, "confusion_matrices")
        os.makedirs(confusion_dir, exist_ok=True)

        # Bin edges for true values
        min_val = torch.min(y_true_part)
        max_val = torch.max(y_true_part)
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Bin true values (labels 1 to n_bins)
        true_bins = np.digitize(y_true_part, bin_edges, right=False)

        # Bin predicted values (0 = below min, 1–n_bins = regular bins, n_bins+1 = above max)
        pred_bin_edges = np.concatenate(([float("-inf")], bin_edges, [float("inf")]))
        pred_bins = np.digitize(y_pred_part, pred_bin_edges, right=False)

        # Compute confusion matrix
        
        cm = confusion_matrix(true_bins, pred_bins, labels=np.arange(1, n_bins + 1))

        # Define tick labels
        xticklabels = [f"<{min_val:.2f}"] + \
                      [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(n_bins)] + \
                      [f">{max_val:.2f}"]
        yticklabels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(n_bins)]

        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=xticklabels, yticklabels=yticklabels)
        plt.xlabel("Predicted Bin")
        plt.ylabel("True Bin")
        plt.title(f"Confusion Matrix of Power Predictions ({part_label}, {title})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(confusion_dir, f"{title}_confusion_matrix_{part_label.lower()}.png")
        plt.savefig(save_path)
        plt.close()

    # Handle real part
    _plot_single_confusion(np.real(y_true_all), np.real(y_pred_all), "Real Part")

    # Handle imaginary part
    _plot_single_confusion(np.imag(y_true_all), np.imag(y_pred_all), "Imaginary Part")


def get_weighted_adjacency_matrix(edge_index, edge_weight, num_nodes=None, symmetric=False):
    """
    Constructs a weighted adjacency matrix from edge_index and edge_weight WITHOUT torch_sparse.

    Args:
        edge_index (LongTensor): [2, num_edges] tensor of edge indices.
        edge_weight (Tensor): [num_edges] or [num_edges, 1] tensor of weights.
        num_nodes (int, optional): Number of nodes. If None, inferred from edge_index.
        symmetric (bool): If True, ensures the adjacency matrix is symmetric.

    Returns:
        adj (Tensor): [num_nodes, num_nodes] dense tensor representing the weighted adjacency matrix.
    """

    if edge_weight.dim() == 2 and edge_weight.size(1) == 1:
        edge_weight = edge_weight.squeeze()
    print('FUCK')
    mask = (edge_index[0] == 153) & (edge_index[1] == 153)

    # Extract edge attribute(s)
    matching_edge_attrs = edge_weight[mask]

    print("Edge attribute(s) for edge ({}, {}):".format(153, 153))
    print(matching_edge_attrs)
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # Optionally make symmetric by adding reverse edges
    if symmetric:
        reversed_edge_index = edge_index[[1, 0]]
        edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    # Initialize adjacency matrix with zeros (complex dtype if needed)
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)

    # Sum weights for duplicate edges
    # To do this efficiently, we can do a loop or a scatter_add on flattened indices

    # Flatten 2D indices to 1D indices for easy aggregation
    flat_indices = edge_index[0] * num_nodes + edge_index[1]

    # Use scatter_add to sum duplicates
    adj_flat = torch.zeros(num_nodes * num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    adj_flat = adj_flat.scatter_add(0, flat_indices, edge_weight)

    # Reshape back to (num_nodes, num_nodes)
    adj = adj_flat.view(num_nodes, num_nodes)

    return adj


# %%
def calc_S_from_output(output, labels, reference, basekV, use_edge_labels=True, P_threshold=0):
    print('Calculating S from output...')
    if use_edge_labels:
        edge_status = labels
    else:
        edge_status = torch.argmax(output[1], dim=1)

    #active_mask = (edge_status == 1)
    #updated_edge_index = reference.edge_index[:, active_mask]
    #updated_edge_attr = reference.edge_attr[active_mask, :]
    updated_edge_index = reference.edge_index
    updated_edge_attr = reference.edge_attr



    if VERBOSE:
        print('UPDATED EDGE ATTRIBUTES')
        print(f'R \n {updated_edge_attr}')

    src, dst = updated_edge_index


    V_kV = output[0][:, 0] + 1j * output[0][:, 1]

    
    V = V_kV / basekV


    updated_edge_attr = updated_edge_attr[:, 0] + 1j * updated_edge_attr[:, 1]
    Y = get_weighted_adjacency_matrix(updated_edge_index, updated_edge_attr, num_nodes=2000, symmetric=False)
    print(Y[827,824])
    print(Y[824,827])
    #YV= Y.to(torch.float) @ V.to(torch.float)
    #print(Y.type)
    #print(V.type)
    YV= Y.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
    S = V * YV.conj()
    print('Full matrix test')
    print(S[827])


    #Y_ij = updated_edge_attr[:, 0] + 1j * updated_edge_attr[:, 1]
    """mask = src < dst
    src = src[mask]
    dst = dst[mask]
    Y_ij = Y_ij[mask]
    V_j = V[dst]
    messages = Y_ij * V_j
    YV = scatter_add(messages, src, dim=0, dim_size=V.shape[0])
    # Find connected admittance values
    connected_mask = (src == 827) | (dst == 827)
    Y_connected = Y_ij[connected_mask]
    print(Y_connected)
    print(f'Y_ij shape: {Y_ij.shape}')
    print(f'V_j shape: {V_j.shape}')
    print(V[827])
    print(YV[827])
    S = V * YV.conj()"""
    S_MVA = S*100
    print(S_MVA[827])
    """
    if P_threshold != 0:
        high_p_buses = np.where(S_MVA.real < P_threshold)[0]

        print(f"\n🧵 Intermediate results for buses with P < {P_threshold} MW:")
        for bus in high_p_buses:
            print(f"\n🔎 Bus {bus}:")
            print(f"  Voltage: {V[bus]:.4f} pu")
            print(f"  Power injection: {S_MVA[bus].real:.4f} + j{S_MVA[bus].imag:.4f} MVA")

            # Find edges where bus is the source
            connected_idxs = np.where(updated_edge_index[0] == bus)[0]
            connected_buses = updated_edge_index[1, connected_idxs]
            print(f"  Connected to buses: {connected_buses}")

            for idx in connected_idxs:
                conn_bus = updated_edge_index[1, idx]
                y_ij = updated_edge_attr[idx]
                voltage_conn = V[conn_bus]
                current = y_ij * (V[bus] - voltage_conn)
                power_flow = V[bus] * np.conj(current)

                print(f"    → Bus {conn_bus}:")
                print(f"      Voltage: {voltage_conn} pu")
                print(f"      Admittance: {y_ij} S")
                print(f"      Current from {bus} to {conn_bus}: {current} A")
                print(f"      Power flow from {bus} to {conn_bus}: {power_flow.real} + j{power_flow.imag} MVA")
    """

    return S



def calc_gridload(S):
    loads = S[S.real < 0]
    grid_load = torch.sum(abs(loads))
    return grid_load

def calc_gridload_filtered(S, threshold=-10):
    loads = S[S.real < 0]
    filtered_loads = loads[loads.real > threshold]
    filtered_grid_load = torch.sum(abs(filtered_loads))
    return filtered_grid_load



def plot_combined_bus_values(S_voltage, S_voltage_true, S_power, S_power_true, title, path, tick_step=20):
    S_voltage_real = S_voltage.real.float()
    S_voltage_imag = S_voltage.imag.float()
    S_voltage_true_real = S_voltage_true.real.float()
    S_voltage_true_imag = S_voltage_true.imag.float()

    S_power_real = S_power.real.float()
    S_power_imag = S_power.imag.float()
    S_power_true_real = S_power_true.real.float()
    S_power_true_imag = S_power_true.imag.float()

    indices = torch.arange(len(S_voltage))
    tick_indices = indices[::tick_step]

    def create_dual_scatter_subplot(fig_title, upper_vals_true, upper_vals_pred, lower_vals_true, lower_vals_pred, ylabel_upper, ylabel_lower, filename_suffix):
        fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=False)

        # Voltage scatter
        axs[0].scatter(upper_vals_true, upper_vals_pred, alpha=0.7)
        axs[0].plot([upper_vals_true.min(), upper_vals_true.max()],
                    [upper_vals_true.min(), upper_vals_true.max()], 'r--', lw=1)
        axs[0].set_ylabel(ylabel_upper)
        axs[0].set_title(fig_title + " - Voltage")
        axs[0].set_xlabel("True")
        #axs[0].set_xlim([upper_vals_true.min(), upper_vals_true.max()])
        #axs[0].set_ylim([upper_vals_true.min(), upper_vals_true.max()])

        # Power scatter
        axs[1].scatter(lower_vals_true, lower_vals_pred, alpha=0.7)
        axs[1].plot([lower_vals_true.min(), lower_vals_true.max()],
                    [lower_vals_true.min(), lower_vals_true.max()], 'r--', lw=1)
        axs[1].set_ylabel(ylabel_lower)
        axs[1].set_title(fig_title + " - Power Injection")
        axs[1].set_xlabel("True")
        #axs[1].set_xlim([lower_vals_true.min(), lower_vals_true.max()])
        #axs[1].set_ylim([lower_vals_true.min(), lower_vals_true.max()])

        plt.tight_layout()
        plt.savefig(f"{path}{title}_{filename_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_dual_bar_difference_subplot(fig_title, upper_vals_diff, lower_vals_diff, ylabel_upper, ylabel_lower, filename_suffix):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axs[0].bar(indices, upper_vals_diff, color='purple')
        axs[0].set_ylabel(ylabel_upper)
        axs[0].set_title(fig_title + " - Voltage")

        axs[1].bar(indices, lower_vals_diff, color='purple')
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel(ylabel_lower)
        axs[1].set_title(fig_title + " - Power Injection")
        axs[1].set_xticklabels([str(i.item()) for i in tick_indices], rotation=90)

        plt.tight_layout()
        plt.savefig(f"{path}{title}_{filename_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Real part scatter
    create_dual_scatter_subplot(
        fig_title=f"{title} - Real Part",
        upper_vals_true=S_voltage_true_real,
        upper_vals_pred=S_voltage_real,
        lower_vals_true=S_power_true_real,
        lower_vals_pred=S_power_real,
        ylabel_upper="Voltage (Real)",
        ylabel_lower="Power (Real)",
        filename_suffix="real"
    )

    # Imaginary part scatter
    create_dual_scatter_subplot(
        fig_title=f"{title} - Imaginary Part",
        upper_vals_true=S_voltage_true_imag,
        upper_vals_pred=S_voltage_imag,
        lower_vals_true=S_power_true_imag,
        lower_vals_pred=S_power_imag,
        ylabel_upper="Voltage (Imag)",
        ylabel_lower="Power (Imag)",
        filename_suffix="imag"
    )

    # Real difference bar plot
    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Real Difference",
        upper_vals_diff=S_voltage_true_real - S_voltage_real,
        lower_vals_diff=S_power_true_real - S_power_real,
        ylabel_upper="Voltage Diff (Real)",
        ylabel_lower="Power Diff (Real)",
        filename_suffix="real_diff"
    )

    # Imaginary difference bar plot
    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Imaginary Difference",
        upper_vals_diff=S_voltage_true_imag - S_voltage_imag,
        lower_vals_diff=S_power_true_imag - S_power_imag,
        ylabel_upper="Voltage Diff (Imag)",
        ylabel_lower="Power Diff (Imag)",
        filename_suffix="imag_diff"
    )




def inspect_high_power_buses(reference, output, labels, basekV, node_labels, P_THRESHOLD=1.0):
    """
    Inspect intermediate results for buses with real power > threshold using PyG-style input.

    Parameters:
    - reference: contains edge_index and edge_attr
    - output: list/tuple where output[0] is voltage (real, imag)
    - labels: edge_status mask (1 = active)
    - basekV: base voltage in kV
    - node_labels: ground truth bus-level data (same format as output)
    - P_THRESHOLD: real power injection threshold (MW)
    """

    edge_status = labels
    active_mask = (edge_status == 1)

    edge_index = reference.edge_index[:, active_mask]
    edge_attr = reference.edge_attr[active_mask, :]

    n_buses = len(output[0])
    V_out = output[0][:, 0] + 1j * output[0][:, 1]
    V_out /= basekV

    V_label = node_labels[0][:, 0] + 1j * node_labels[0][:, 1]
    V_label /= basekV

    src, dst = edge_index
    Y_ij = edge_attr[:, 0] + 1j * edge_attr[:, 1]

    # --- Model-based power injection ---
    I_out = scatter_add(Y_ij * (V_out[src] - V_out[dst]), src, dim=0, dim_size=n_buses)
    S_out = V_out * np.conj(I_out)

    # --- Label-based power injection ---
    I_label = scatter_add(Y_ij * (V_label[src] - V_label[dst]), src, dim=0, dim_size=n_buses)
    S_label = V_label * np.conj(I_label)

    S_label *= 100
    S_out *= 100

    print('Hello')
    print(S_label)
    print(S_out)

    # --- Inspect buses over threshold ---
    high_p_buses = np.where(S_out.real < P_THRESHOLD)[0]

    print(f"\n🧵 Intermediate results for buses with P < {P_THRESHOLD} MW:")
    for bus in high_p_buses:
        print(f"\n🔎 Bus {bus}:")
        print(f"  [Output] Voltage: {V_out[bus]:.4f} pu")
        print(f"  [Output] Power injection: {S_out[bus].real:.4f} + j{S_out[bus].imag:.4f} MVA")

        print(f"  [Label ] Voltage: {V_label[bus]:.4f} pu")
        print(f"  [Label ] Power injection: {S_label[bus].real:.4f} + j{S_label[bus].imag:.4f} MVA")

        # Edges where bus is the source
        connected_idxs = np.where(edge_index[0] == bus)[0]
        connected_buses = edge_index[1, connected_idxs]
        print(f"  Connected to buses: {connected_buses}")
        for idx in connected_idxs:
            conn_bus = edge_index[1, idx]
            adm = Y_ij[idx]

            # Output-based
            Vb_out, Vc_out = V_out[bus], V_out[conn_bus]
            Ibc_out = adm * (Vb_out - Vc_out)
            Sbc_out = Vb_out * np.conj(Ibc_out)

            # Label-based
            Vb_label, Vc_label = V_label[bus], V_label[conn_bus]
            Ibc_label = adm * (Vb_label - Vc_label)
            Sbc_label = Vb_label * np.conj(Ibc_label)

            print(f"    → Bus {conn_bus}:")
            print(f"      Admittance: {adm} S")

            print(f"      [Output] Voltage: {Vc_out} pu")
            print(f"      [Output] Current: {Ibc_out} A")
            print(f"      [Output] Power flow: {Sbc_out.real} + j{Sbc_out.imag} MVA")

            print(f"      [Label ] Voltage: {Vc_label} pu")
            print(f"      [Label ] Current: {Ibc_label} A")
            print(f"      [Label ] Power flow: {Sbc_label.real} + j{Sbc_label.imag} MVA")


# %%
#data= scipy.io.loadmat('/home/tohlinger/Documents/hi-accf-ml/raw/pwsdata.mat')
#node_data = data['clusterresult_'][0,0][2] 


#Load matlab data for basekV and comparison
f = h5py.File('/home/tohlinger/Documents/hi-accf-ml/raw/clusterresults_1.mat', 'r')
bus_data_ref = f['clusterresult_']['bus']
ref = bus_data_ref[4, 0]
dereferenced_data = f[ref]
matlab_node_data = [dereferenced_data[()]]
matlab_node_data = torch.tensor(np.array(matlab_node_data).squeeze()).transpose(0, 1)

basekV = matlab_node_data[:, 9]
matlab_Vm = matlab_node_data[:, 7]
matlab_Va = matlab_node_data[:, 8]
matlab_Vm_kV = matlab_Vm * basekV
matlab_V_complex = matlab_Vm_kV * (np.cos(np.deg2rad(matlab_Va)) + 1j * np.sin(np.deg2rad(matlab_Va)))

if VERBOSE:
    print(f'MATLAB DATA ')
    print(f'Matlab P \n {matlab_node_data[:, 2]}')
    print(f'Matlab Q \n {matlab_node_data[:, 3]}')
    print(f'Matlab V\n{matlab_V_complex}\n')

#get the normalized (raw) labels and output
node_labels_normalized = labels[0]
node_output_normalized = output[0]

if VERBOSE:
    print(f'Normalized Pytorch Data')
    print('LABELS')
    print(f'Normalized P \n {node_labels_normalized[:, 0]}')
    print(f'Normalized Q \n {node_labels_normalized[:, 1]}')
    print('\nOUTPUT')
    print(f'Normalized P \n {node_output_normalized[:, 0]}')
    print(f'Normalized Q \n {node_output_normalized[:, 1]}\n')



#denormalize output, labels and edge_attributes
denormalized_output = output[0]*(torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
denormalized_output = (denormalized_output, output[1])
node_labels = labels[0] * (torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
print(f'node_labels (denormailzed) shape: {node_labels[:,0].shape}')
static.edge_attr[:, 0] = static.edge_attr[:, 0] * (min_max['max_values']['edge_attr'][0] - min_max['min_values']['edge_attr'][0]) + min_max['min_values']['edge_attr'][0]
static.edge_attr[:, 1] = static.edge_attr[:, 1] * (min_max['max_values']['edge_attr'][1] - min_max['min_values']['edge_attr'][1]) + min_max['min_values']['edge_attr'][1]

if VERBOSE:
    print(f'\nEdge Status')
    print(f'Edge Status \n {labels[1].sum()}')

N_INSTANCES = output[0].shape[0] // 2000
grid_load = []
grid_load_true = []
grid_load_filtered = []
S_all = []
S_true_all = []

for i in range(int(N_INSTANCES)):
    S = calc_S_from_output((denormalized_output[0][i*2000:(i+1)*2000],), labels[1][i], static, basekV, use_edge_labels=True, P_threshold=0)*100
    print(f'Instance {i}: Calculated S shape: {S.shape}')
    S_true = calc_S_from_output((node_labels[i*2000:(i+1)*2000],), labels[1][i], static, basekV, use_edge_labels=True, P_threshold=0)*100
    #S = calc_S_from_output((denormalized_output[0][i*2000:(i+1)*2000],), labels[1], static, basekV, use_edge_labels=True, P_threshold=0)
    #S_true = calc_S_from_output((node_labels[i*2000:(i+1)*2000],), labels[1], static, basekV, use_edge_labels=True)
    grid_load.append(calc_gridload(S * 100).item())
    grid_load_true.append(calc_gridload(S_true * 100).item())
    grid_load_filtered.append(calc_gridload_filtered(S * 100, -10*100).item())
    S_all.append(S)
    S_true_all.append(S_true)

#inspect_high_power_buses(static, (output[0][:2000], ), labels[1][0], basekV, (node_labels[:2000],), P_THRESHOLD=-50)   
#inspect_high_power_buses(static_edge_attr, (output[0][:2000], ), labels[1], basekV, (node_labels[:2000],), P_THRESHOLD=-50)   

print(f'Grid Load: {grid_load} MVA')
print(f'Grid Load True: {grid_load_true} MVA')
print(f'Grid Load Filtered: {grid_load_filtered} MVA')

if VERBOSE:
    print('Min S_true real')
    print(min(S_true_all[0].real))
    print('Max S_true real')
    print(max(S_true_all[0].real))
    print('Min S_true imag' )
    print(min(S_true_all[0].imag))
    print('Max S_true imag')
    print(max(S_true_all[0].imag))
    print(f'Calculated S: {S_all[0].shape}')
    print(f'Calculated S True: {S_true_all[0].shape}')
    

# Save to file
results = {
    'S': S_all,
    'S_true': S_true_all,
    'grid_load': grid_load,
    'grid_load_true': grid_load_true
}

with open(PATH + 'grid_load_results.pkl', 'wb') as f:
    pkl.dump(results, f)




# %%
# Plot histograms
#Plot bus values of first instance

plot_combined_bus_values(torch.complex(output[0][:2000, 0].float(), output[0][:2000, 1].float()), torch.complex(labels[0][:2000, 0].float(), labels[0][:2000, 1].float()), 
                         S_all[0][:2000],S_true_all[0][:2000], 'bus_combined', PATH, tick_step=50)

# Group predictions and truths

pred_by_deg_real, true_by_deg_real, pred_by_deg_imag, true_by_deg_imag = collect_predictions_by_degree(S_all, S_true_all, labels[1], static)

# Plot per degree

for deg in sorted(true_by_deg_real.keys()):
    if deg != 0 and deg <13:
        print(f'DEG: {deg}')
        plot_confusion_matrix_power_predictions(
            y_true_all=torch.complex(torch.tensor(true_by_deg_real[deg]), torch.tensor(true_by_deg_imag[deg])),
            y_pred_all=torch.complex(torch.tensor(pred_by_deg_real[deg]), torch.tensor(pred_by_deg_imag[deg])),
            title=f"deg{deg}_real",
            path=PATH
        )

r2_real, mse_real, r2_imag, mse_imag = compute_and_plot_metrics(
    pred_by_deg_real, true_by_deg_real,
    pred_by_deg_imag, true_by_deg_imag, PATH
)


S_true_all_ = [item for sublist in S_true_all for item in sublist]
S_all_ = [item for sublist in S_all for item in sublist]
S_true_all = torch.stack(S_true_all_)
S_all = torch.stack(S_all_)

plot_confusion_matrix_power_predictions(S_true_all.reshape(-1), S_all.reshape(-1), 'power_predictions', PATH, n_bins=10)




# --- HISTOGRAM: Normalized output[0] vs. labels[0] ---
plt.figure(figsize=(8, 4))
#plt.hist(output[0][:2000].detach().numpy().flatten(), bins=50, alpha=0.6, label='Predicted', color='blue')
plt.hist(labels[0][:2000].detach().numpy().flatten(), bins=50, alpha=0.6, label='True', color='orange')
plt.xlabel('Normalized Value')
plt.ylabel('Frequency')
plt.title('Normalized Output vs. Labels (First 2000 Buses)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_norm_output_vs_labels.png', dpi=300, bbox_inches='tight')
plt.close()

# --- HISTOGRAM: Unnormalized output[0] vs. unnormalized labels[0] ---
plt.figure(figsize=(8, 4))
#plt.hist(denormalized_output[0][:2000].detach().numpy().flatten(), bins=50, alpha=0.6, label='Predicted', color='blue')
plt.hist(node_labels[:2000].detach().numpy().flatten(), bins=50, alpha=0.6, label='True', color='orange')
plt.xlabel('Unnormalized Value')
plt.ylabel('Frequency')
plt.title('Unnormalized Output vs. Labels (First 2000 Buses)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_unnorm_output_vs_labels.png', dpi=300, bbox_inches='tight')
plt.close()

# --- HISTOGRAM: S vs. S_true for first instance ---
#S0 = S_all[0].detach().numpy()
#S_true0 = S_true_all[0].detach().numpy()
S0 = S_all[0].detach().numpy()
S_true0 = S_true_all[0].detach().numpy()

plt.figure(figsize=(8, 4))
#plt.hist(S0.real, bins=50, alpha=0.6, label='Predicted Real', color='blue')
plt.hist(S_true0.real, bins=100, alpha=0.6, label='True Real', color='orange')
plt.xlabel('S_real [p.u.]')
plt.ylabel('Frequency')
plt.title('Real Part of S (First 2000 Buses)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_S_real_vs_true.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 4))
#plt.hist(S0.imag, bins=50, alpha=0.6, label='Predicted Imag', color='blue')
plt.hist(S_true0.imag, bins=100, alpha=0.6, label='True Imag', color='orange')
plt.xlabel('S_imag [p.u.]')
plt.ylabel('Frequency')
plt.title('Imaginary Part of S (First 2000 Buses)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_S_imag_vs_true.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 4))
#plt.hist(S0.imag, bins=50, alpha=0.6, label='Predicted Imag', color='blue')
plt.hist(node_labels[:,0].numpy(), bins=100, alpha=0.6, label='True Imag', color='orange')
plt.xlabel('V true real [kV]')
plt.ylabel('Frequency')
plt.title('V true real (instance 111)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_V_true_realkV.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 4))
#plt.hist(S0.imag, bins=50, alpha=0.6, label='Predicted Imag', color='blue')
plt.hist(node_labels[:,1].numpy(), bins=100, alpha=0.6, label='True Imag', color='orange')
plt.xlabel('V true imag [kV]')
plt.ylabel('Frequency')
plt.title('V true imag (instance 111)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_V_true_imagkV.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 4))
#plt.hist(S0.imag, bins=50, alpha=0.6, label='Predicted Imag', color='blue')
plt.hist((node_labels[:,0]/basekV).numpy(), bins=100, alpha=0.6, label='True Imag', color='orange')
plt.xlabel('V true real [p.u.]')
plt.ylabel('Frequency')
plt.title('V true real (instance 111)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_V_true_real.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 4))
#plt.hist(S0.imag, bins=50, alpha=0.6, label='Predicted Imag', color='blue')
plt.hist((node_labels[:,1]/basekV).numpy(), bins=100, alpha=0.6, label='True Imag', color='orange')
plt.xlabel('V true imag [p.u.]')
plt.ylabel('Frequency')
plt.title('V true imag (instance 111)')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'hist_V_true_imag.png', dpi=300, bbox_inches='tight')
plt.close()




plt.figure(figsize=(8, 4))
all_data = np.concatenate([grid_load, grid_load_true])
bins = np.histogram_bin_edges(all_data, bins=50)

plt.hist(grid_load, bins=bins, alpha=0.6, label='Predicted', color='blue')
plt.hist(grid_load_true, bins=bins, alpha=0.6, label='True', color='orange')
plt.xlabel('Grid Load [MVA]')
plt.ylabel('Frequency')
plt.title('Histogram of Grid Load')
plt.legend()
plt.tight_layout()
plt.savefig(PATH + 'grid_load_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
# %%
