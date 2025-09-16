# %%
import torch
import scipy
import numpy as np
import h5py
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import f1_score
import time
from sklearn.metrics import r2_score, mean_squared_error


def labels_vs_mat(
    matlab_bus_data,
    matlab_gen_data,
    S_pred, path,
):
    """
    Compare predicted power injections (S_pred) with true injections from MATLAB data.

    Args:
        matlab_bus_data (np.ndarray or list): Array of bus objects with .bus_i, .Pd, .Qd
        matlab_gen_data (np.ndarray or list): Array of generator objects with .bus, .Pg, .Qg
        S_pred (torch.Tensor): Complex tensor of predicted power injections from ML model (in MVA)

    Returns:
        S_true (torch.Tensor): Ground truth power injections as complex tensor
    """
    # Step 1: Map arbitrary bus IDs to 0-based continuous indices
    all_bus_ids = np.unique(np.concatenate((matlab_bus_data[:, 0], matlab_gen_data[:, 0])))
    bus_id_to_index = {int(bus_id): idx for idx, bus_id in enumerate(all_bus_ids)}
    num_buses = len(all_bus_ids)

    # Step 2: Initialize arrays
    Pd = np.zeros(num_buses)
    Qd = np.zeros(num_buses)
    Pg = np.zeros(num_buses)
    Qg = np.zeros(num_buses)

    # Step 3: Fill demand (Pd, Qd)
    for row in matlab_bus_data:
        bus_id = int(row[0])
        idx = bus_id_to_index[bus_id]
        Pd[idx] = row[2]
        Qd[idx] = row[3]

    # Step 4: Fill generation (Pg, Qg)
    for row in matlab_gen_data:
        bus_id = int(row[0])
        idx = bus_id_to_index[bus_id]
        #print(row[1])
        if matlab_bus_data[idx,1] !=4:
            Pg[idx] += row[1]  # sum multiple generators at same bus
            Qg[idx] += row[2]

    # Ground truth S (complex)
    S_true_np = (Pg - Pd) + 1j * (Qg - Qd)
    S_matlab = torch.tensor(S_true_np, dtype=torch.complex64)
    print('S_matlab')
    print(S_matlab.shape)
    print('S_pred')
    print(S_pred.shape)
    # --- Plot Real Power ---
    plt.figure(figsize=(6, 6))
    plt.scatter(S_matlab.real, S_pred.real, alpha=0.5, label='P (Real)')
    plt.plot([-100, 100], [-100, 100], 'k--', label='Ideal')
    plt.xlabel('Matlab P injection [MW]')
    plt.ylabel('Labels P injection [MW]')
    plt.title('Real Power Injection Comparison')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{path}matlab_vs_labels_real.png", dpi=300, bbox_inches='tight')

    # --- Plot Reactive Power ---
    plt.figure(figsize=(6, 6))
    plt.scatter(S_matlab.imag, S_pred.imag, alpha=0.5, label='Q (Imag)', color='orange')
    plt.plot([-100, 100], [-100, 100], 'k--', label='Ideal')
    plt.xlabel('Matlab Q injection [MVAr]')
    plt.ylabel('Labels Q injection [MVAr]')
    plt.title('Reactive Power Injection Comparison')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{path}matlab_vs_labels_imag.png", dpi=300, bbox_inches='tight')
    return S_matlab


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



def create_b_edge_attr(bus_IDs, branch_data, edge_index, b):
    bus_id_map = {int(bus_id): idx for idx, bus_id in enumerate(bus_IDs)}
    from_buses_raw = branch_data[:, 0].astype(int)
    to_buses_raw   = branch_data[:, 1].astype(int)

    line_to_shunt = {}
    for fb_raw, tb_raw, b in zip(from_buses_raw, to_buses_raw, b):
        fb = bus_id_map[fb_raw]
        tb = bus_id_map[tb_raw]
        i, j = sorted((fb, tb))
        line_to_shunt[(i, j)] = 1j * b / 2  # jb/2

# Step 4: Assign jb/2 to each edge in edge_index
    shunt_attr = torch.zeros(edge_index.shape[1], dtype=torch.cfloat)
    for k in range(edge_index.shape[1]):
        i = edge_index[0, k].item()
        j = edge_index[1, k].item()
        key = tuple(sorted((i, j)))
        if key in line_to_shunt:
            shunt_attr[k] = torch.tensor(line_to_shunt[key], dtype=torch.cfloat)
    return shunt_attr


def calculate_S(output, labels, Y, edge_index, b, use_edge_labels):
    #calc S
    edge_index = static_data.edge_index
    #edge_weight = static_data.edge_attr
    #edge_weight = edge_weight.type(torch.complex64)
    S_all = []
    degrees_all = []
    print('Used edge label shape')
    for i in range(int(len(output[0])/2000)):
        if use_edge_labels:
            print('Used edge labels')
            Y_instance = build_Y_matrix_from_predictions(Y.clone(), labels[1][i], edge_index, b)
        else:
            print('Used output labels')
            instance_output = (output[0], torch.argmax(output[1][i*7064:(i+1)*7064], dim=1))  # shape: [batch_size]
            Y_instance = build_Y_matrix_from_predictions(Y.clone(), instance_output[1], edge_index, b)
        print('Y_instance: ', Y_instance)
        print('Y_isntance first line', Y_instance[0,63])
        V = output[0][i*2000:(i+1)*2000,:].float()
        print('V in calc S: ', V)
        V = torch.complex(V[:, 0], V[:, 1])
        #V = V/basekV
        print('V p.u. during calc S:', V)

        YV= Y_instance.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
        S = V * YV.conj()
        print(max(S.real))

        #get degrees for plotting
        degree_instance = (Y_instance.abs() > 0).sum(dim=1)
        degree_instance = degree_instance - (Y_instance.diag().abs() > 0).long()  # remove diagonal self-admittance if needed

        S_all.append(S)
        degrees_all.append(degree_instance)
    return torch.tensor(np.array(S_all)), Y_instance, torch.tensor(np.array(degrees_all))

def build_Y_matrix_from_predictions(Y_, edge_predictions, edge_index, b):
    Y= Y_.clone()

    inactive_edges = torch.where(edge_predictions.flatten() == 0)[0]
    for idx in inactive_edges:
        i, j = edge_index[:,idx]
        y_ij = Y[i, j]
   
        if i!=j:
            Y[i, i] += y_ij - b[idx]
            Y[i, j] = 0

    Y[abs(Y.real)<0.001] = 0j
    return torch.complex(torch.tensor(Y.real), torch.tensor(Y.imag)).type(torch.complex64) 


def calc_gridload(S):
    loads = S[S.real < 0]
    grid_load = torch.sum(abs(loads))
    return grid_load


def plot_combined_bus_values(S_voltage, S_voltage_true, S_power, S_power_true, title, path, basekV, tick_step=20, degrees = None):
    print('Predicted Voltage kV:', S_voltage)
    print('Labels Voltage kV:', S_voltage_true)
    print('basekV:', basekV)

    #for i in range(int(len(S_voltage)/2000)):
    #    print('ENTERED')
    #    S_voltage[i*2000:(i+1)*2000] = S_voltage[i*2000:(i+1)*2000] / basekV.float()
    #    S_voltage_true[i*2000:(i+1)*2000] = S_voltage_true[i*2000:(i+1)*2000] / basekV.float()
    print('Predicted Voltage p.u.:', S_voltage)
    print('Labels Voltage p.u.:', S_voltage_true)
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

    def create_dual_scatter_subplot(fig_title, upper_vals_true, upper_vals_pred, lower_vals_true, lower_vals_pred, ylabel_upper, ylabel_lower, filename_suffix, degrees=degrees):
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
        if degrees is None: 
            sc = axs[1].scatter(lower_vals_true, lower_vals_pred, c=upper_vals_true, alpha=0.7)
            cbar = fig.colorbar(sc, ax=axs[1])
            cbar.set_label("Voltage (True)")
        else:
            sc = axs[1].scatter(lower_vals_true, lower_vals_pred, c=degrees, alpha=0.7)
            cbar = fig.colorbar(sc, ax=axs[1])
            cbar.set_label("Node Degree")
        axs[1].plot([lower_vals_true.min(), lower_vals_true.max()],
                    [lower_vals_true.min(), lower_vals_true.max()], 'r--', lw=1)
        axs[1].set_ylabel(ylabel_lower)
        axs[1].set_title(fig_title + " - Power Injection")
        axs[1].set_xlabel("True")
        #axs[1].set_ylim([-200, 200])
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
        filename_suffix="real",
        degrees=degrees
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
        filename_suffix="imag",
        degrees=degrees
    )

    # Real difference bar plot
    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Real Voltage ",
        upper_vals_diff=S_voltage_true_real,
        lower_vals_diff=S_voltage_real ,
        ylabel_upper="Voltage true [p.u.]  (Real)",
        ylabel_lower="Voltage predicted[p.u.]  (Real)",
        filename_suffix="voltage_real"
    )

    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Real Power",
        upper_vals_diff=S_power_true_real,
        lower_vals_diff=S_power_real,
        ylabel_upper="Power true [p.u.] (Real)",
        ylabel_lower="Power predicted [p.u.] (Real)",
        filename_suffix="power_real"
    )

    

    # Imaginary difference bar plot
    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Imaginary Voltage",
        upper_vals_diff=S_voltage_true_imag,
        lower_vals_diff=S_voltage_imag,
        ylabel_upper="Voltage true [p.u.] (Imag)",
        ylabel_lower="Voltage predicted [p.u.] (Imag)",
        filename_suffix="voltage_imag"
    )

        # Imaginary difference bar plot
    create_dual_bar_difference_subplot(
        fig_title=f"{title} - Imaginary Difference",
        upper_vals_diff=S_power_true_imag,
        lower_vals_diff=S_power_imag,
        ylabel_upper="Power true [p.u.] (Imag)",
        ylabel_lower="Power predicted [p.u.] (Imag)",
        filename_suffix="power_imag"
    )




load_start = time.time()
DENORMALIZE = False
N_PLOTS = 2
PATH = '/home/tohlinger/HUI/Documents/hi-accf-ml/'
output = torch.load(PATH + 'results/test_output_final.pt')
labels = torch.load(PATH + 'results/test_labels_final.pt')
static_data = torch.load(PATH +'processed/data_static.pt')
pwsdata = scipy.io.loadmat(PATH + 'raw/pwsdata.mat')
with open('/home/tohlinger/HUI/Documents/hi-accf-ml/min_max_PU_test.pkl', 'rb') as f:
    min_max = pkl.load(f)   #load min_max values for denormalization

print('Load time (s):', time.time() - load_start)

basekV = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,9] )
bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
Y = torch.tensor(pwsdata['clusterresult_'][0,0][10] )   #in p.u.
branch_data = pwsdata['clusterresult_'][0,0][4]

edge_index = static_data.edge_index
b = torch.tensor(branch_data[:,4] ) #in p.u.

edge_attr_b_start = time.time()
b = create_b_edge_attr(bus_IDs, branch_data, edge_index, b) #in p.u.
print('Calculating b as Edge attr time (s):', time.time() - edge_attr_b_start)


#denormalize output, labels and edge_attributes
denormalize_start = time.time()
if DENORMALIZE: denormalized_output = output[0]*(torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
else:           denormalized_output = output[0]
denormalized_output = (denormalized_output, output[1])
del output
print('Denormalization time output (s):', time.time() - denormalize_start)
denormalize_start = time.time()
if DENORMALIZE: node_labels = labels[0]* (torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
else:           node_labels = labels[0]
denormalized_labels = (node_labels, labels[1])
print('Denormalization time labels (s):', time.time() - denormalize_start)
print('denormalized output: ', denormalized_output[0])
print('denormalized labels: ', denormalized_labels[0])
calculate_s_start = time.time()
S_true_all, Y_true, degrees_true = calculate_S(denormalized_labels, labels, Y, edge_index, b, use_edge_labels=True)
print('Calculating S true time (s):', time.time() - calculate_s_start)
calculate_s_start = time.time()
S_pred_all, Y_pred, degrees_pred = calculate_S(denormalized_output, labels, Y, edge_index, b, use_edge_labels=True)
print('Degrees: ', degrees_pred)
print('Calculating S pred time (s):', time.time() - calculate_s_start)

print('S_true_all:', S_true_all)
print('S_pred_all:', S_pred_all)

print('Equal Y?: ', torch.equal(Y_true, Y_pred))
print('Y_true sum:', torch.sum(Y_true))
print('Y_pred sum:', torch.sum(Y_pred))
edge_output = torch.argmax(denormalized_output[1], dim=1)
print('Edge output shape:', edge_output.shape)
print('Labels edge shape:', labels[1].shape)
print('F1 of edges:', f1_score(1-edge_output.cpu().numpy(), 1-labels[1].flatten().cpu().numpy().squeeze(), average='macro'))

MSE_real = mean_squared_error(S_true_all.real, S_pred_all.real)
MSE_imag = mean_squared_error(S_true_all.imag, S_pred_all.imag)
R2_imag = r2_score(S_true_all.imag, S_pred_all.imag)
R2_real = r2_score(S_true_all.real, S_pred_all.real)
print(f'MSE: {MSE_real}')
print(f'R2: {R2_real}')
print(f'MSE imag: {MSE_imag}')
print(f'R2 imag: {R2_imag}')

grid_load_true = []
grid_load_pred = []
for i in range(len(S_true_all)):
    grid_load_true.append(calc_gridload(S_true_all[i]))
    grid_load_pred.append(calc_gridload(S_pred_all[i]))

for i in range(N_PLOTS):
    plotting_start = time.time()
    plot_combined_bus_values(torch.complex(denormalized_output[0][i*2000:(i+1)*2000, 0].float(), denormalized_output[0][i*2000:(i+1)*2000, 1].float()), 
                             torch.complex(node_labels[i*2000:(i+1)*2000, 0].float(), node_labels[i*2000:(i+1)*2000, 1].float()), 
                            S_pred_all.flatten()[i*2000:(i+1)*2000], S_true_all.flatten()[i*2000:(i+1)*2000], f'bus_combined{i}', 
                            PATH, basekV, tick_step=50, degrees = degrees_pred.flatten()[i*2000:(i+1)*2000])
    print('Plotting time (s):', time.time() - plotting_start)
#plot_combined_bus_values(output_pu, labels_pu,
#                         S_pred_all.flatten(), S_true_all.flatten(), 'bus_combined', PATH, tick_step=50)





#ONLY FOR TESTING AGAINST MATLAB REMOVE LATER
"""
file = h5py.File('/home/tohlinger/HUI/Documents/hi-accf-ml/raw__/clusterresults_111.mat', 'r')
Y_matlab = []
Y_matlab_ref = file['clusterresult_']['Ybus_ext']
ref = Y_matlab_ref[47,0]
dereferenced_data = file[ref]
Y_matlab.append(dereferenced_data[()]) 
real = np.vectorize(lambda x: x[0])(Y_matlab)
imag = np.vectorize(lambda x: x[1])(Y_matlab)

# Step 2: Convert to PyTorch tensors
real_tensor = torch.tensor(real, dtype=torch.float32) #in p.u.
imag_tensor = torch.tensor(imag, dtype=torch.float32) #in p.u.

# Step 3: Combine into a complex tensor
Y_matlab = torch.complex(real_tensor, imag_tensor)[0].type(torch.complex64) 
print('Y_matlab==Y_true:', torch.equal(Y_matlab, Y_true))
for i in range(len(Y_matlab)):
    for j in range(len(Y_matlab)):
        if Y_matlab[i,j] != Y_true[i,j]:
            print('Difference at index ', i, ': ', Y_matlab[i,i], ' vs ', Y_true[i,i])

bus_data_ref = file['clusterresult_']['bus']
ref = bus_data_ref[47, 0]
dereferenced_data = file[ref]
matlab_node_data = [dereferenced_data[()]]
matlab_node_data = torch.tensor(np.array(matlab_node_data).squeeze()).transpose(0, 1)
gen_data_ref = file['clusterresult_']['gen']
ref = gen_data_ref[47, 0]
dereferenced_data = file[ref]
matlab_gen_data = [dereferenced_data[()]]
matlab_gen_data = torch.tensor(np.array(matlab_gen_data).squeeze()).transpose(0, 1)

#END TESTING AGAINST MATLAB

S_matlab = labels_vs_mat(matlab_node_data, matlab_gen_data, S_true_all*100, PATH)
"""
# %%
