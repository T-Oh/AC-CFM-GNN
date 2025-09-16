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

        S_all.append(S)
    return torch.tensor(np.array(S_all)), Y_instance

def build_Y_matrix_from_predictions(Y_, edge_predictions, edge_index, b):
    print('Used edge labels/outputs')
    print(edge_predictions[100:1000])
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


def plot_combined_bus_values(S_voltage, S_voltage_true, S_power, S_power_true, title, path, basekV, tick_step=20):
    print('Predicted Voltage kV:', S_voltage)
    print('Labels Voltage kV:', S_voltage_true)
    print('basekV:', basekV)

    for i in range(int(len(S_voltage)/2000)):
        print('ENTERED')
        S_voltage[i*2000:(i+1)*2000] = S_voltage[i*2000:(i+1)*2000] / basekV.float()
        S_voltage_true[i*2000:(i+1)*2000] = S_voltage_true[i*2000:(i+1)*2000] / basekV.float()
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




load_start = time.time()

PATH = '/home/tohlinger/HUI/Documents/hi-accf-ml/'
output = torch.load(PATH + 'results/output_final.pt')
labels = torch.load(PATH + 'results/labels_final.pt')
static_data = torch.load(PATH +'processed/data_static.pt')
pwsdata = scipy.io.loadmat(PATH + 'raw/pwsdata.mat')
with open('/home/tohlinger/HUI/Documents/hi-accf-ml/min_max_ASnoCl.pkl', 'rb') as f:
    min_max = pkl.load(f)   #load min_max values for denormalization

print('Load time (s):', time.time() - load_start)

basekV = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,9] )
bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
Y = torch.tensor(pwsdata['clusterresult_'][0,0][10] )
branch_data = pwsdata['clusterresult_'][0,0][4]

edge_index = static_data.edge_index
b = torch.tensor(branch_data[:,4] )

edge_attr_b_start = time.time()
b = create_b_edge_attr(bus_IDs, branch_data, edge_index, b)
print('Calculating b as Edge attr time (s):', time.time() - edge_attr_b_start)


#denormalize output, labels and edge_attributes
denormalize_start = time.time()
denormalized_output = output[0].clone()#*(torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
denormalized_output = (denormalized_output, output[1])
#del output
print('Denormalization time output (s):', time.time() - denormalize_start)
denormalize_start = time.time()
node_labels = labels[0].clone() #* (torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
denormalized_labels = (node_labels, labels[1])
print('Denormalization time labels (s):', time.time() - denormalize_start)
print('denormalized output: ', denormalized_output[0])
print('denormalized labels: ', denormalized_labels[0])
#output_pu = torch.complex(denormalized_output[0][:, 0]/basekV.float(), denormalized_output[0][:, 1]/basekV.float())
#labels_pu = torch.complex(node_labels[:, 0].float(), node_labels[:, 1].float()) / basekV.float()
calculate_s_start = time.time()
S_true_all, Y_true = calculate_S(denormalized_labels, labels, Y, edge_index, b, use_edge_labels=True)
print('Calculating S true time (s):', time.time() - calculate_s_start)
calculate_s_start = time.time()
S_pred_all, Y_pred = calculate_S(denormalized_output, labels, Y, edge_index, b, use_edge_labels=True)
print('Calculating S pred time (s):', time.time() - calculate_s_start)

print('S_true_all:', S_true_all)
print('S_pred_all:', S_pred_all)

print('Equal Y?: ', torch.equal(Y_true, Y_pred))
print('Y_true sum:', torch.sum(Y_true))
print('Y_pred sum:', torch.sum(Y_pred))
edge_output = torch.argmax(denormalized_output[1], dim=1)
print('Edge output shape:', edge_output.shape)
print('Labels edge shape:', labels[1].shape)
print('F1 of edges:', f1_score(1-edge_output.cpu().numpy(), 1-labels[1].cpu().numpy().squeeze(), average='macro'))

grid_load_true = []
grid_load_pred = []
for i in range(len(S_true_all)):
    grid_load_true.append(calc_gridload(S_true_all[i]))
    grid_load_pred.append(calc_gridload(S_pred_all[i]))


plotting_start = time.time()
plot_combined_bus_values(torch.complex(denormalized_output[0][:, 0].float(), denormalized_output[0][:, 1].float()), torch.complex(node_labels[:, 0].float(), node_labels[:, 1].float()), 
                         S_pred_all.flatten(), S_true_all.flatten(), 'bus_combined', PATH, basekV, tick_step=50)
print('Plotting time (s):', time.time() - plotting_start)
#plot_combined_bus_values(output_pu, labels_pu,
#                         S_pred_all.flatten(), S_true_all.flatten(), 'bus_combined', PATH, tick_step=50)


#ONLY FOR TESTING AGAINST MATLAB REMOVE LATER
file = h5py.File('/home/tohlinger/HUI/Documents/hi-accf-ml/raw__/clusterresults_111.mat', 'r')
Y_matlab = []
Y_matlab_ref = file['clusterresult_']['Ybus_ext']
ref = Y_matlab_ref[47,0]
dereferenced_data = file[ref]
Y_matlab.append(dereferenced_data[()]) 
real = np.vectorize(lambda x: x[0])(Y_matlab)
imag = np.vectorize(lambda x: x[1])(Y_matlab)

# Step 2: Convert to PyTorch tensors
real_tensor = torch.tensor(real, dtype=torch.float32)
imag_tensor = torch.tensor(imag, dtype=torch.float32)

# Step 3: Combine into a complex tensor
Y_matlab = torch.complex(real_tensor, imag_tensor)[0].type(torch.complex64) 

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

S_matlab = labels_vs_mat(matlab_node_data, matlab_gen_data, S_true_all, PATH)

# %%
