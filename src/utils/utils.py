#from sklearn.base import r2_score
import time
import torch
import json
import os
import torch.nn
import matplotlib.pyplot as plt
import warnings

from torch_geometric.data import Data
from torch import scatter_add
from sys import argv

from utils.loss_functions import weighted_loss_label, state_loss, physics_loss, state_loss_power_injection


def check_config_conflicts(cfg):
    assert not (cfg['crossvalidation'] and cfg['study::run']),  'can only run a study or the crossvalidation not both'
    assert not (cfg['data'] == 'DC' and cfg['stormsplit']>0),   'Stormsplit can only be used with AC data'
    assert not (cfg['edge_attr'] == 'multi' and cfg['model'] == 'TAG'), 'TAG can only be used with Y as edge_attr not with multi'
    assert not (cfg['data'] == 'LDTSF' and cfg['task'] == 'NodeReg'),   'LDTSF Only works with GraphReg and GraphClass'
    assert not (cfg['data'] == 'LDTSF' and cfg['model'] != 'lstm'),     'LDTSF Only works with lstm as model'
    assert not (cfg['data'] == 'AC' and cfg['task'] == 'GraphClass'),   'None of the models working with AC data has GraphClass implemented' 
    assert not (cfg['model'] not in ['GATLSTM', 'TAGLSTM', 'MLPLSTM'] and cfg['task'] in ['StateRegPI', 'StateReg']), 'StateReg and StateRegPI only work with LSTM models'
    if cfg['process'] and os.path.exists(os.path.join(cfg['dataset::path'], 'processed')):
        assert not os.listdir(os.path.join(cfg['dataset::path'], 'processed')), 'Processed directory is not empty, please clear it before processing again to avoid conflicts. If you want to keep the processed data, set process to false and run with the desired model and task.'
    if cfg['normalize'] and os.path.exists(os.path.join(cfg['dataset::path'], 'normalized')):
        assert not os.listdir(os.path.join(cfg['dataset::path'], 'normalized')), 'Normalized directory is not empty, please clear it before normalizing again to avoid conflicts. If you want to keep the normalized data, set normalize to false and run with the desired model and task.'
    if cfg['data'] == 'LSTM' and not cfg['model'] in ['GATLSTM', 'TAGLSTM', 'MLPLSTM']: 
        warnings.warn("Using Time Series data with a model that has no recurrent layers, this should only be done for processing of LSTM data", UserWarning)
    

def save_output(output, labels, test_output, test_labels, name=""):
    with open("results/" + "output"+name+".pt", "wb") as f:
        torch.save(output, f)
    with open("results/" + "labels"+name+".pt", "wb") as f:
        torch.save(labels, f)
    with open("results/" + "test_output"+name+".pt", "wb") as f:
        torch.save(test_output, f)
    with open("results/" + "test_labels"+name+".pt", "wb") as f:
        torch.save(test_labels, f)


def choose_criterion(task, use_weighted_loss_label, weighted_loss_factor, cfg, device):
            # Init Criterion
        if cfg['physics_loss']:
            criterion = physics_loss(cfg['pl_w1'], cfg['pl_w2'], cfg['pl_w3'], device)
        
        elif task in ['GraphClass', 'typeIIClass']:
            criterion = torch.nn.CrossEntropyLoss()
        elif use_weighted_loss_label:
            criterion = weighted_loss_label(
            factor=torch.tensor(weighted_loss_factor))
        elif task == 'StateReg':
            criterion = state_loss(weighted_loss_factor, cfg['dataset::path'], cfg['weight_nodes_by_centrality'], cfg['weight_nodes_by_voltage_threshold'], cfg['voltage_weight'])
        elif task == 'StateRegPI':
            criterion = state_loss_power_injection(cfg, weighted_loss_factor, cfg['PI_factor'], device)
        elif task == 'NodeReg':
            criterion = torch.nn.MSELoss(reduction='mean')
        return criterion

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim > 0 else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_params(path, params, ID=None):
    del params['mask_probs']
    with open(os.path.join(path,'results/', ID+'_params.json'), 'w') as f:
        json.dump(params, f, default=tensor_to_serializable, indent=4)

def multiclass_classification(output, labels, N_bins):

    output = output.reshape(-1)
    labels = labels.reshape(-1)
    N_nodes = len(output)
    labelclasses = torch.zeros(N_bins+1)
    outputclasses = torch.zeros(N_bins+1)
    matrix = torch.zeros([N_bins+1, N_bins+1])
    for node in range(N_nodes):
        for i in range(N_bins+1):
            if output[node] <= (1/N_bins)*i:
                outputclasses[i] += 1
                break;
        for j in range(N_bins+1):
            if labels[node] <= (1/N_bins)*j:
                labelclasses[j] += 1
                matrix[j,i] += 1
                break;
    return outputclasses, labelclasses, matrix
    
def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()**2
    return total_norm**0.5


def create_data_from_prediction(predicted_node_features, predicted_edge_status, reference, node_labels, edge_labels):
    """
    Create a new Data object from predicted node features and edge status.
    This function assumes that the reference Data object is the static data (i.e. 7064 edges)
    and that the edge status is a binary tensor indicating the presence of edges.
    """

    # Mask for active lines
    active_mask = (predicted_edge_status == 0)  # [E], boolean

    # Apply mask to get filtered topology and admittances
    updated_edge_index = reference.edge_index[:, active_mask]       # [2, E']
    updated_edge_attr  = reference.edge_attr[active_mask]           # [E']


    # Get source and target nodes of each edge
    src, dst = updated_edge_index  # [E], [E]

    V = predicted_node_features[:,0] + 1j * predicted_node_features[:,1]  # [E], complex tensor
    # Get edge admittances Y_ij
    #Y_ij = updated_edge_attr       # [E], complex tensor
    Y_ij = updated_edge_attr[:, 0] + 1j * updated_edge_attr[:, 1]  # shape [E], dtype=torch.complex64 or complex128


    # V_j values at source nodes (i.e., neighbor voltages)
    V_j = V[src]           # [E]
    print('V_j:', V_j.shape)
    print('Y_ij:', Y_ij.shape)
    # Compute message: Y_ij * V_j
    messages = Y_ij * V_j  # [E]


    # Aggregate incoming messages at target node (i.e., YV at each node)
    YV = scatter_add(messages, dst, dim=0, dim_size=V.shape[0])  # [N]
    S = V * YV.conj()  # [N]
    new_node_features = torch.stack((predicted_node_features[:,0], predicted_node_features[:,1], S.real, S.imag), dim=1)  # [N, 2]


    new_data = Data(
        x=new_node_features,  # shape [num_nodes, num_node_features]
        edge_index = updated_edge_index,
        edge_attr = updated_edge_attr,
        node_labels = node_labels,  # shape [num_nodes, num_node_features]
        edge_labels = edge_labels,  # shape [num_edges, num_edge_features]
        # optionally include dummy y or node_labels if needed
    )
    return new_data


# Helper to safely read argv[i]
def get_arg(i, default=1):
    try:
        return int(argv[i])
    except (IndexError, ValueError):
        return default




def zhu_perform_bus_check(bus_type, bus_type_post, P1_net, Q1_net, Vm, Va, P2_net, Q2_net, Vm2, Va2):
    path_plots = 'plots_fixed/'
    time_index = int(time.time() * 1000.0) % 10000

    if torch.any(bus_type_post - bus_type != 0):
        print('BUS TYPE HAS CHANGED')

    tolerance = 1e-4

    PQ_bus_P_diff = torch.mul(P2_net - P1_net, bus_type[:, 0])
    PQ_bus_Q_diff = torch.mul(Q2_net - Q1_net, bus_type[:, 0])
    PV_bus_P_diff = torch.mul(P2_net - P1_net, bus_type[:, 1])
    PV_bus_V_diff = torch.mul(Vm2 - Vm, bus_type[:, 1])
    slack_bus_V_diff = torch.mul(Vm2 - Vm, bus_type[:, 2])
    slack_bus_angle_diff = torch.mul(Va2 - Va, bus_type[:, 2])

    # Checking whether P are constant for PQ buses
    P1_plot = torch.mul(P1_net, bus_type[:, 0])
    if not torch.allclose(PQ_bus_P_diff, torch.zeros_like(PQ_bus_P_diff), atol=tolerance):
        fig00, ax00 = plt.subplots()
        ax00.plot(P1_plot.numpy(), label='P1 times T_PQ')
        ax00.plot(PQ_bus_P_diff.to('cpu').numpy(), label='P2-P1 times T_PQ')
        ax00.legend()
        fig00.savefig(path_plots + f'P_for_PQ_{time_index}.png', bbox_inches='tight')

    # Checking whether Q are constant for PQ buses
    Q1_plot = torch.mul(Q1_net, bus_type[:, 0])
    if not torch.allclose(PQ_bus_Q_diff, torch.zeros_like(PQ_bus_Q_diff), atol=tolerance):
        fig01, ax01 = plt.subplots()
        ax01.plot(Q1_plot.numpy(), label='Q1 times T_PQ')
        ax01.plot(PQ_bus_Q_diff.to('cpu').numpy(), label='Q2-Q1 times T_PQ')
        ax01.legend()
        fig01.savefig(path_plots + f'Q_for_PQ_{time_index}.png', bbox_inches='tight')

    # Checking whether P are constant for PV buses
    P1_PV_plot = torch.mul(P1_net, bus_type[:, 1])
    if not torch.allclose(PV_bus_P_diff, torch.zeros_like(PV_bus_P_diff), atol=tolerance):
        fig10, ax10 = plt.subplots()
        ax10.plot(P1_PV_plot.numpy(), label='P1 times T_PV')
        ax10.plot(PV_bus_P_diff.to('cpu').numpy(), label='P2-P1 times T_PV')
        ax10.legend()
        fig10.savefig(path_plots + f'P_for_PV_{time_index}.png', bbox_inches='tight')

    # Checking whether Vm is constand for PV buses
    Vm_plot = torch.mul(Vm, bus_type[:, 1])
    if not torch.allclose(PV_bus_V_diff, torch.zeros_like(PV_bus_V_diff), atol=tolerance):
        fig11, ax11 = plt.subplots()
        ax11.plot(Vm_plot.numpy(), label='Vm')
        ax11.plot(PV_bus_V_diff.to('cpu').numpy(), label='Vm2-Vm times T_PV')
        ax11.legend()
        fig11.savefig(path_plots + f'Vm_for_PV_{time_index}.png', bbox_inches='tight')

    # Checking whether Vm is constand for slack buses
    if torch.any(slack_bus_V_diff != 0):
        fig112, ax112 = plt.subplots()
        ax112.plot(Vm_plot.numpy(), label='Vm')
        ax112.plot(slack_bus_V_diff.to('cpu').numpy(), label='Vm2-Vm times T_slack')
        ax112.legend()
        fig112.savefig(path_plots + f'Vm_for_slack_{time_index}.png', bbox_inches='tight')

    # Checking whether voltage phase angle is constant for slack buses
    if torch.any(slack_bus_angle_diff != 0):
        fig110, ax110 = plt.subplots()
        ax110.plot(Vm_plot.numpy(), label='Va')
        ax110.plot(slack_bus_angle_diff.to('cpu').numpy(), label='Va2-Va times T_slack')
        ax110.legend()
        fig110.savefig(path_plots + f'Va_for_slack_{time_index}.png', bbox_inches='tight')


def check_s_y_relation(self, node_data_post, edge_data_post, gen_data_post):
    admittance_matrix = torch.tensor(edge_data_post)

    '''CHECKING WHETHER S=V(YV)*'''
    '''CALCULATE POST S COMPLEX'''
    P2_demand = torch.tensor(node_data_post[:, 2])
    Q2_demand = torch.tensor(node_data_post[:, 3])

    gen_features_post = self.get_gen_features(gen_data_post, node_data_post)
    P2_generated = gen_features_post[:, 0]
    Q2_generated = gen_features_post[:, 1]

    P2_net = (P2_generated - P2_demand) / 100  # dividing by base 100 to convert to per unit system
    Q2_net = (Q2_generated - Q2_demand) / 100  # dividing by base 100 to convert to per unit system

    S_fromPQ_post = torch.complex(P2_net, Q2_net).unsqueeze(1)

    '''CALCULATE POST V COMPLEX'''
    Vm_post = torch.tensor(node_data_post[:, 7])
    Va_post = torch.tensor(node_data_post[:, 8])

    Va_radians_post = torch.deg2rad(Va_post)

    Vreal_post = Vm_post * torch.cos(Va_radians_post)
    Vimag_post = Vm_post * torch.sin(Va_radians_post)

    Vcomplex_post = torch.complex(Vreal_post, Vimag_post)

    node_labels = torch.view_as_real(Vcomplex_post)

    Vcomplex_post_unsqueezed = Vcomplex_post.unsqueeze(1)

    S_fromVY_post = torch.mul(Vcomplex_post_unsqueezed,
                                torch.conj(torch.matmul(admittance_matrix, Vcomplex_post_unsqueezed)))

    if torch.allclose(S_fromPQ_post, S_fromVY_post, atol=1e-7):
        print('S_fromPQ_post and S_fromVY_post all are close SUCCESS')
    else:
        print('S_fromPQ_post and S_fromVY_post ARE NOT CLOSE FAILURE')

    '''FINISHED CHECKING WHETHER S=V(YV)*'''
    '''CONSTRUCT EDGE ATTR FROM ADMITTANCE MATRIX'''


