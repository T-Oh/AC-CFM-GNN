# %%
import torch
import scipy
import numpy as np
import h5py
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_sparse import coalesce



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
    print(S_matlab)
    print('S_pred')
    print(S_pred)
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





def calculate_S(output, labels, Y, edge_index, use_edge_labels):
    for i in range(int(len(output[0])/2000)):
        if use_edge_labels:
            Y_instance = build_Y_matrix_from_predictions(Y, labels[1][7064*i:7064*(i+1)], edge_index)
        else:
            Y_instance = build_Y_matrix_from_predictions(Y, output[1][7064*i:7064*(i+1)], edge_index)
    #TESTING
    file = h5py.File('/home/tohlinger/HUI/Documents/hi-accf-ml/raw/clusterresults_1.mat', 'r')
    Y_matlab = []
    Y_matlab_ref = file['clusterresult_']['Ybus_ext']
    ref = Y_matlab_ref[43,0]
    dereferenced_data = file[ref]
    Y_matlab.append(dereferenced_data[()]) 
    
    real = np.vectorize(lambda x: x[0])(Y_matlab)
    imag = np.vectorize(lambda x: x[1])(Y_matlab)

    # Step 2: Convert to PyTorch tensors
    real_tensor = torch.tensor(real, dtype=torch.float32)
    imag_tensor = torch.tensor(imag, dtype=torch.float32)

    # Step 3: Combine into a complex tensor
    Y_matlab = torch.complex(real_tensor, imag_tensor)[0].type(torch.complex64) 

    diff_mask = Y_matlab != Y_instance

    # Get the indices where values differ
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    for i in range(len(diff_indices)):
        print(f"Difference at index {diff_indices[i]}: {Y_matlab[diff_indices[i,0], diff_indices[i,1]]} vs {Y_instance[diff_indices[i,0], diff_indices[i,1]]}")

    #print('Y matlab')
    #print(Y_matlab)
    #print(Y_instance)
    print(torch.sum(Y_matlab-Y_instance))
    print(torch.equal(Y_matlab, Y_instance))
    return Y_instance, Y_matlab


def build_Y_matrix_from_predictions(Y_, edge_predictions, edge_index, b):
    Y= Y_.clone()
    inactive_edges = torch.where(edge_predictions[0] == 0)[0]
    #print(f'Inactive edges: {inactive_edges}')
    #print('Labels in build Y matrix')
    #print(edge_predictions[0,edge_index[0]==1977])
    #print(f'According edges {edge_index[:,edge_index[0]==1977]}')
    for idx in inactive_edges:
        i, j = edge_index[:,idx]
        y_ij = Y[i, j]
        #if i==1977 or j == 1977:
            #print(f'Edge {i}-{j} is inactive, setting Y[{i},{j}] and Y[{j},{i}] to 0')
            #print(f'y_ij {y_ij}')
            #print(f'Y_ii before {Y[i,i]}')
   
        if i!=j:
            Y[i, i] += y_ij - b[idx]
            Y[i, j] = 0

        #if i==1977 or j == 1977:
            #print(f'Y_ij after {Y[i,j]}')
            #print(f'Y_ii after {Y[i,i]}')

    Y[abs(Y.real)<0.001] = 0j
    return torch.complex(torch.tensor(Y.real), torch.tensor(Y.imag)).type(torch.complex64) 



PATH = '/home/tohlinger/HUI/Documents/hi-accf-ml/'
output = torch.load(PATH + 'results/output_final.pt')
labels = torch.load(PATH + 'results/labels_final.pt')
static_data = torch.load(PATH+'processed/data_static.pt')
pwsdata = scipy.io.loadmat(PATH + 'raw/pwsdata.mat')
with open('/home/tohlinger/HUI/Documents/hi-accf-ml/min_max_ASnoCl.pkl', 'rb') as f:
    min_max = pkl.load(f)   #load min_max values for denormalization

basekV = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,9] )
bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
Y = torch.tensor(pwsdata['clusterresult_'][0,0][10] )
branch_data = pwsdata['clusterresult_'][0,0][4]

edge_index = static_data.edge_index
b = torch.tensor(branch_data[:,4] )

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



#ONLY FOR TESTING AGAINST MATLAB REMOVE LATER
file = h5py.File('/home/tohlinger/HUI/Documents/hi-accf-ml/raw__/clusterresults_1.mat', 'r')
Y_matlab = []
Y_matlab_ref = file['clusterresult_']['Ybus_ext']
ref = Y_matlab_ref[43,0]
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
ref = bus_data_ref[43, 0]
dereferenced_data = file[ref]
matlab_node_data = [dereferenced_data[()]]
matlab_node_data = torch.tensor(np.array(matlab_node_data).squeeze()).transpose(0, 1)
gen_data_ref = file['clusterresult_']['gen']
ref = gen_data_ref[43, 0]
dereferenced_data = file[ref]
matlab_gen_data = [dereferenced_data[()]]
matlab_gen_data = torch.tensor(np.array(matlab_gen_data).squeeze()).transpose(0, 1)

#END TESTING AGAINST MATLAB


#calc S
edge_index = static_data.edge_index
edge_weight = static_data.edge_attr
edge_weight = edge_weight.type(torch.complex64)
use_edge_labels=True
for i in range(int(len(output[0])/2000)):
    if use_edge_labels:
        Y_instance = build_Y_matrix_from_predictions(Y.clone(), labels[1][7064*i:7064*(i+1)], edge_index, shunt_attr )
    else:
        Y_instance = build_Y_matrix_from_predictions(Y.clone(), output[1][7064*i:7064*(i+1)], edge_index, shunt_attr)





#denormalize output, labels and edge_attributes
denormalized_output = output[0]*(torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])
denormalized_output = (denormalized_output, output[1])
node_labels = labels[0] * (torch.tensor(min_max['max_values']['node_labels']) - torch.tensor(min_max['min_values']['node_labels'])) + torch.tensor(min_max['min_values']['node_labels'])

V = denormalized_output[0]
V = torch.complex(V[:, 0], V[:, 1])
V = V/basekV
print(f'V shape {V}')
print(f'Y shape {Y_instance}')
YV= Y_instance.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
S = V * YV.conj()

S_matlab = labels_vs_mat(matlab_node_data, matlab_gen_data, S*100, PATH)
#TESTING
diff_mask = abs(Y_matlab - Y_instance)>0.2

# Get the indices where values differ
diff_indices = torch.nonzero(diff_mask, as_tuple=False)
count_differences = 0
for i in range(len(diff_indices)):
    print(f"Difference at index {diff_indices[i]}: {Y_matlab[diff_indices[i,0], diff_indices[i,1]]} vs {Y_instance[diff_indices[i,0], diff_indices[i,1]]}")
    count_differences += 1
print(f'Total differences found: {count_differences}')

#print('Y matlab')
#print(Y_matlab)
#print(Y_instance)
print(torch.sum(Y_matlab-Y_instance))
print(torch.equal(Y_matlab, Y_instance))

#Investigate remaining differences in S_imag
diff_mask = abs(S_matlab.imag - S.imag*100)/S_matlab.imag > 0.1 
diff_indices = torch.nonzero(diff_mask, as_tuple=False)
count_differences = 0
for i in range(len(diff_indices)):
    if abs(S[diff_indices[i,0]].imag) > 0.1:
        print(f"Difference in S_imag at index {diff_indices[i]}: {S_matlab[diff_indices[i,0]].imag} vs {S[diff_indices[i,0]].imag*100}")
        count_differences += 1
print(f'Total differences in S_imag found: {count_differences}')

# %%
