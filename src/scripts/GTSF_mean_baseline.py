import torch
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def calculate_mean_baselines(labels):
    """
    Calculates three different mean baselines: 1. the overall mean, 2. the mean per node, 3. the mean per instance
    Input:
        labels: tensor of shape (num_instances, num_nodes) with S=P+iQ (torch.complex)
    Output:
        overall_mean: tensor of shape (num_nodes,) with the overall mean value
        mean_per_node: tensor of shape (num_nodes,) with the mean value per node
        mean_per_instance: tensor of shape (num_instances,) with the mean value per instance
    """
    overall_mean = labels.mean(dim=0).mean()
    mean_per_node = labels.mean(dim=0)
    mean_per_instance = labels.mean(dim=1)
    overall_mean = torch.zeros(2000) + overall_mean
    return overall_mean, mean_per_node, mean_per_instance



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


def calculate_S(labels, Y, edge_index, b):
    #calc S
    edge_index = static_data.edge_index
    S_all = []
    for i in range(int(len(labels[0])/2000)):

        Y_instance = build_Y_matrix_from_predictions(Y.clone(), labels[1][i], edge_index, b)

        V = labels[0][i*2000:(i+1)*2000,:].float()
        V = torch.complex(V[:, 0], V[:, 1])

        YV= Y_instance.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
        S = V * YV.conj()

        S_all.append(S)


    return torch.tensor(np.array(S_all))

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


#SETTINGS
PATH = '/home/tohlinger/HUI/Documents/hi-accf-ml/'
RESULT_FOLDER = os.path.join(PATH, 'results/')
PATH_STATIC_DATA = '/home/tohlinger/HUI/Documents/hi-accf-ml/processed/data_static.pt'

#LOAD DATA
static_data = torch.load(PATH_STATIC_DATA)  #using edge_index to calculate b
pwsdata = scipy.io.loadmat(os.path.join(PATH, 'raw/pwsdata.mat'))   #using original matlab data to calculate Y
train_labels = torch.load(os.path.join(RESULT_FOLDER, 'labels.pt')) #used to calculate the means
test_labels = torch.load(os.path.join(RESULT_FOLDER, 'test_labels.pt')) #used to calculate losses
test_output = torch.load(os.path.join(RESULT_FOLDER, 'test_output.pt')) #not used here

#extract relevant data from matlab file
bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
Y = torch.tensor(pwsdata['clusterresult_'][0,0][10] )
branch_data = pwsdata['clusterresult_'][0,0][4]
b = torch.tensor(branch_data[:,4] ) #in p.u.

edge_index = static_data.edge_index

b = create_b_edge_attr(bus_IDs, branch_data, edge_index, b) #in p.u.

S_train = calculate_S(train_labels, Y, edge_index, b)

overall_mean, mean_per_node, mean_per_instance = calculate_mean_baselines(S_train)

#CALCULATE LOSSES
S_test = calculate_S(test_labels, Y, edge_index, b)
S_test_pred = calculate_S(test_output, Y, edge_index, b)

S_test_absolute = torch.abs(S_test)

overall_MSE = mean_squared_error(S_test_absolute.flatten(), abs(overall_mean).repeat(len(S_test)).flatten())
overall_R2_real = r2_score(S_test.flatten().real, overall_mean.repeat(len(S_test)).real)
overall_R2_imag = r2_score(S_test.flatten().imag, overall_mean.repeat(len(S_test)).imag)
per_node_MSE = mean_squared_error(S_test_absolute.flatten(), abs(mean_per_node).repeat(len(S_test)).flatten())
per_node_R2_real = r2_score(S_test.flatten().real, mean_per_node.repeat(len(S_test)).flatten().real)
per_node_R2_imag = r2_score(S_test.flatten().imag, mean_per_node.repeat(len(S_test)).flatten().imag)
#per_instance_MSE = mean_squared_error(S_test_absolute.flatten(), abs(mean_per_instance).repeat_interleave(2000).flatten())
#per_instance_R2_real = r2_score(S_test.flatten().real, mean_per_instance.repeat_interleave(2000).flatten().real)
#per_instance_R2_imag = r2_score(S_test.flatten().imag, mean_per_instance.repeat_interleave(2000).flatten().imag)

print(f'Overall Mean Baseline: MSE={overall_MSE}, R2_real={overall_R2_real}, R2_imag={overall_R2_imag}')
print(f'Mean per Node Baseline: MSE={per_node_MSE}, R2_real={per_node_R2_real}, R2_imag={per_node_R2_imag}')
#print(f'Mean per Instance Baseline: MSE={per_instance_MSE}, R2_real={per_instance_R2_real}, R2_imag={per_instance_R2_imag}')




# --- PLOTTING ---

# --- SETTINGS ---
num_instances_to_plot = 10  # <--- control how many instances to plot
num_instances_to_plot = min(num_instances_to_plot, len(S_test))

# Convert to numpy for easier plotting
S_test_np = S_test.detach().cpu().numpy()
S_test_pred_np = S_test_pred.detach().cpu().numpy()
mean_per_node_np = mean_per_node.detach().cpu().numpy()
#mean_per_instance_np = mean_per_instance.detach().cpu().numpy()
overall_mean_np = overall_mean.detach().cpu().numpy()

# --- PLOT 1: S_test_pred vs S_test ---
for i in range(num_instances_to_plot):
    true_vals = S_test_np[i]
    pred_vals = S_test_pred_np[i]
    #inst_mean = mean_per_instance_np[i]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"S_test_pred vs S_test — Instance {i}")

    # Real part
    axs[0].scatter(true_vals.real, pred_vals.real, s=10, alpha=0.5, label="Predicted")
    axs[0].axhline(y=overall_mean_np.real.mean(), color="red", linestyle="--", label="Overall mean")
    #axs[0].axhline(y=inst_mean.real.mean(), color="orange", linestyle=":", label="Instance mean")
    axs[0].set_xlabel("True Real(S)")
    axs[0].set_ylabel("Predicted Real(S)")
    axs[0].set_title("Real part")
    axs[0].legend()

    # Imag part
    axs[1].scatter(true_vals.imag, pred_vals.imag, s=10, alpha=0.5, label="Predicted")
    axs[1].axhline(y=overall_mean_np.imag.mean(), color="red", linestyle="--", label="Overall mean")
    #axs[1].axhline(y=inst_mean.imag.mean(), color="orange", linestyle=":", label="Instance mean")
    axs[1].set_xlabel("True Imag(S)")
    axs[1].set_ylabel("Predicted Imag(S)")
    axs[1].set_title("Imag part")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_FOLDER, f'S_test_pred_vs_S_test_instance_{i}.png'))


# --- PLOT 2: mean_per_node vs S_test ---
for i in range(num_instances_to_plot):
    true_vals = S_test_np[i]
    #inst_mean = mean_per_instance_np[i]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Mean per Node vs S_test — Instance {i}")

    # Real part
    axs[0].scatter(true_vals.real, mean_per_node_np.real, s=10, alpha=0.5, label="Nodewise mean")
    axs[0].axhline(y=overall_mean_np.real.mean(), color="red", linestyle="--", label="Overall mean")
    #axs[0].axhline(y=inst_mean.real.mean(), color="orange", linestyle=":", label="Instance mean")
    axs[0].set_xlabel("True Real(S)")
    axs[0].set_ylabel("Nodewise Mean Real(S)")
    axs[0].set_title("Real part")
    axs[0].legend()

    # Imag part
    axs[1].scatter(true_vals.imag, mean_per_node_np.imag, s=10, alpha=0.5, label="Nodewise mean")
    axs[1].axhline(y=overall_mean_np.imag.mean(), color="red", linestyle="--", label="Overall mean")
    #axs[1].axhline(y=inst_mean.imag.mean(), color="orange", linestyle=":", label="Instance mean")
    axs[1].set_xlabel("True Imag(S)")
    axs[1].set_ylabel("Nodewise Mean Imag(S)")
    axs[1].set_title("Imag part")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_FOLDER, f'mean_per_node_vs_S_test_instance_{i}.png'))


