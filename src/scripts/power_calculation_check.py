# %%
import torch
from torch_scatter import scatter_add
import scipy
import pickle as pkl
import numpy as np
import h5py
import matplotlib.pyplot as plt



# %%
# Verbosity flag
#def main():
VERBOSE = True
#Loading data
PATH = '/home/tohlinger/HUI/Documents/hi-accf-ml/'
output_raw = torch.load(PATH + 'results/output_final.pt')
labels_raw = torch.load(PATH + 'results/labels_final.pt')
static = torch.load(PATH + 'processed/data_static.pt') #load static for original edge attributes and indices
with open('/home/tohlinger/HUI/Documents/hi-accf-ml/min_max_ASnoCl.pkl', 'rb') as f:
    min_max = pkl.load(f)   #load min_max values for denormalization
large_discrepancy_factor = 10  # Define a factor to determine large discrepancies
num_to_invest = 1
#Print raw data
print(f'output_raw: {output_raw[0][:20]}')
print(f'labels_raw: {labels_raw[0][:20]}')
print(f'static.edge_attr: {static.edge_attr[:20]}')

#Denormalization
denormalized_output, denormalized_labels, denormalized_edge_attr = denormalize(output_raw, labels_raw, static, min_max)
print(f'denormalized_output: {denormalized_output[0][:20]}')
print(f'denormalized_labels: {denormalized_labels[0][:20]}')
print(f'denormalized_edge_attr: {denormalized_edge_attr[:20]}')

#Load matlab data
matlab_node_data = load_mat_file(PATH+'raw/clusterresults_111.mat', type='h5py')
basekV = matlab_node_data[:, 9]
matlab_Vm = matlab_node_data[:, 7]
matlab_Va = matlab_node_data[:, 8]
matlab_Vm_kV = matlab_Vm * basekV

#Calculate S from output and labels
S_output = calc_S_from_output(denormalized_output, denormalized_labels[1], static, basekV=basekV, use_edge_labels=True, P_threshold=0, VERBOSE=VERBOSE)*100
S_true = calc_S_from_output(denormalized_labels, denormalized_labels[1], static, basekV=basekV, use_edge_labels=True, P_threshold=0, VERBOSE=VERBOSE)*100
print(f'S_output: {S_output[:20]}')
print(f'S_true: {S_true[:20]}')

#Calculate Apparent Power
apparent_power_output = torch.sqrt(S_output.real**2 + S_output.imag**2)
apparent_power_true = torch.sqrt(S_true.real**2 + S_true.imag**2)
print(f'apparent_power_output: {apparent_power_output[:20]}')
print(f'apparent_power_true: {apparent_power_true[:20]}')


# %%
#Find values with large discrepancies and values which are very close and print the corresponding intermediate results
count = 0
count_close = 0
#Variables for investigation
Y_ij_real = []
Y_ij_imag = []
V_i_pred = None
V_i_true = None
V_j_pred_real = []
V_j_pred_imag = []
V_j_true_real = []
V_j_true_imag = []
for i in range(len(apparent_power_output)):
    print(i)
    if count >= num_to_invest:
        break
    if apparent_power_output[i] > apparent_power_true[i]*large_discrepancy_factor:
        print(f"\n🔍 Large discrepancy found at index {i}:")
        print(f"  Output apparent power: {apparent_power_output[i]} MVA")
        print(f"  True apparent power: {apparent_power_true[i]} MVA")
        print(f"  Output S: {S_output[i]}")
        print(f"  True S: {S_true[i]}")
        print(f"  Output voltage: {denormalized_output[0][i]}")
        print(f"  True voltage: {denormalized_labels[0][i]}")
        print(f"  Base kV: {basekV[i]}")
        V_i_pred = torch.complex(denormalized_output[0][i][0]/basekV[i], denormalized_output[0][i][1]/basekV[i])
        V_i_true = torch.complex(denormalized_labels[0][i][0]/basekV[i], denormalized_labels[0][i][1]/basekV[i])
        S_i_true = S_true[i]
        S_i_output = S_output[i]
        
        # Find neighbors of node i
        mask_src = (static.edge_index[0] == i)
        mask_tgt = (static.edge_index[1] == i)

        neighbors_src = static.edge_index[1][mask_src]
        neighbors_tgt = static.edge_index[0][mask_tgt]
        neighbors = torch.cat([neighbors_src, neighbors_tgt]).unique()

        print(f"  Neighbors of bus {i}: {neighbors.tolist()}")
        for j in neighbors:
            j = j.item()
            V_j_pred_real.append(denormalized_output[0][j][0]/basekV[j])
            V_j_pred_imag.append(denormalized_output[0][j][1]/basekV[j])
            V_j_true_real.append(denormalized_labels[0][j][0]/basekV[j])
            V_j_true_imag.append(denormalized_labels[0][j][1]/basekV[j])

            # Get admittance between i and j
            adm_mask_ij = ((static.edge_index[0] == i) & (static.edge_index[1] == j))
            print(denormalized_edge_attr[adm_mask_ij])
            Y_ij_real.append(denormalized_edge_attr[adm_mask_ij][0,0])
            Y_ij_imag.append(denormalized_edge_attr[adm_mask_ij][0,1])

            print(f"    🔗 Neighbor {j}:")
            print(f"      Voltage output: {denormalized_output[0][j]}")
            print(f"      Voltage true  : {denormalized_labels[0][j]}")
            print(f"      Admittance Y_ij: {Y_ij}")
        count += 1
        V_j_pred = torch.stack([torch.tensor(V_j_pred_real), torch.tensor(V_j_pred_imag)], dim=1)
        V_j_pred = torch.complex(V_j_pred[:,0], V_j_pred[:,1])
        V_j_true = torch.stack([torch.tensor(V_j_true_real), torch.tensor(V_j_true_imag)], dim=1)
        V_j_true = torch.complex(V_j_true[:,0], V_j_true[:,1])
        Y_ij = torch.stack([torch.tensor(Y_ij_real), torch.tensor(Y_ij_imag)], dim=1)
        Y_ij = torch.complex(Y_ij[:,0], Y_ij[:,1])

    elif apparent_power_output[i] < apparent_power_true[i] * 1.1 and apparent_power_output[i] > apparent_power_true[i] * 0.9 and count_close < num_to_invest:
        print(f"\n🔍 Large discrepancy found at index {i}:")
        print(f"  Output apparent power: {apparent_power_output[i]} MVA")
        print(f"  True apparent power: {apparent_power_true[i]} MVA")
        print(f"  Output S: {S_output[i]}")
        print(f"  True S: {S_true[i]}")
        print(f"  Output voltage: {denormalized_output[0][i]}")
        print(f"  True voltage: {denormalized_labels[0][i]}")
        print(f"  Base kV: {basekV[i]}")
        # Find neighbors of node i
        mask_src = (static.edge_index[0] == i)
        mask_tgt = (static.edge_index[1] == i)

        neighbors_src = static.edge_index[1][mask_src]
        neighbors_tgt = static.edge_index[0][mask_tgt]
        neighbors = torch.cat([neighbors_src, neighbors_tgt]).unique()

        print(f"  Neighbors of bus {i}: {neighbors.tolist()}")
        for j in neighbors:
            j = j.item()
            # Get admittance between i and j
            adm_mask_ij = ((static.edge_index[0] == i) & (static.edge_index[1] == j))
                            
            Y_ij = denormalized_edge_attr[adm_mask_ij]

            print(f"    🔗 Neighbor {j}:")
            print(f"      Voltage output: {denormalized_output[0][j]}")
            print(f"      Voltage true  : {denormalized_labels[0][j]}")
            print(f"      Admittance Y_ij: {Y_ij}")
        count_close += 1


# %%
def denormalize(output, labels, static, min_max):
    denormalized_output = output[0]*(torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
    denormalized_output = (denormalized_output, output[1])
    node_labels = labels[0] * (torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
    denormalized_labels = (node_labels, labels[1])

    static.edge_attr[:, 0] = static.edge_attr[:, 0] * (torch.tensor(min_max['max_values']['edge_attr'][0]) - torch.tensor(min_max['min_values']['edge_attr'][0])) + torch.tensor(min_max['min_values']['edge_attr'][0])
    static.edge_attr[:, 1] = static.edge_attr[:, 1] * (torch.tensor(min_max['max_values']['edge_attr'][1]) - torch.tensor(min_max['min_values']['edge_attr'][1])) + torch.tensor(min_max['min_values']['edge_attr'][1])

    return denormalized_output, denormalized_labels, static.edge_attr


def load_mat_file(file_path, type='h5py'):
    if type == 'h5py':
        f = h5py.File(file_path, 'r')
        bus_data_ref = f['clusterresult_']['bus']
        ref = bus_data_ref[4, 0]
        dereferenced_data = f[ref]
        matlab_node_data = [dereferenced_data[()]]
        matlab_node_data = torch.tensor(np.array(matlab_node_data).squeeze()).transpose(0, 1)
    else:
        data= scipy.io.loadmat(file_path)
        matlab_node_data = data['clusterresult_'][0,0][2] 
    return matlab_node_data




def calc_S_from_output(output, edge_labels, reference, basekV, use_edge_labels=True, P_threshold=0, VERBOSE=True):
    if use_edge_labels:
        edge_status = edge_labels
    else:
        edge_status = torch.argmax(output[1], dim=1)

    active_mask = (edge_status == 1)
    print(f'edge_index: {reference.edge_index[:20]}')
    print(f'edge_attr: {reference.edge_attr[:20]}')
    updated_edge_index = reference.edge_index[:, active_mask]
    updated_edge_attr = reference.edge_attr[active_mask, :]

    if VERBOSE:
        print('UPDATED EDGE ATTRIBUTES')
        print(f'R \n {updated_edge_attr}')

    src, dst = updated_edge_index
    V_kV = output[0][:, 0] + 1j * output[0][:, 1]

    
    V = V_kV / basekV

    Y_ij = updated_edge_attr[:, 0] + 1j * updated_edge_attr[:, 1]
    V_j = V[dst]
    print(V_j.shape)
    print(Y_ij.shape)
    messages = Y_ij * V_j
    YV = scatter_add(messages, src, dim=0, dim_size=V.shape[0])
    S = V * YV.conj()
    S_MVA = S*100

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
    return S

# %%
if __name__ == "__main__":
    main()