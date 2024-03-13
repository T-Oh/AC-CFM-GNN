from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_undirected
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
import os
from ray import tune

#from tabulate import tabulate
"""
A lot of functions here have been (and still can be) used for analyzing a big variety of stuff within the data or the models. But these functions
are not called during the normal operation of the program. HOWEVER this does not go for all functions in this module.
Here is the list of the functions necessary for the normal operation:
    setup_search_space
    setup_params_from_search_space
    setup_params
    weighted_loss_var
    weighted_loss_label

"""

def from_scipy_coo(A):
    """
    Converts scipy adjacency coo matrix to pytorch gemometric format.
    """
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def to_torch(data):
    """
    Converts network to pytorch_geometric.Data with
    node features active and reactive power, voltage magnitude and phase angle.
    The target value is the active power.

    Input:
        data (dict) : Dictionary containing a node feature matrix,
            adjacency matrix, and edge feature matrix with
            keys "node_features", "adjacency", "edge_features"
    Return:
        (torch_geometric.Data)
    """
    x = torch.tensor(data["node_features"].T, dtype=torch.float64)

    edge_index = from_scipy_coo(data["adjacency"])
    edge_weights = torch.tensor(data["edge_features"], dtype=torch.float64)
    edge_index, edge_weights = to_undirected(edge_index, edge_attr= edge_weights, num_nodes= x.size(0))

    y = torch.tensor(data["y"], dtype =torch.float64).view(-1,1)

    del data["node_features"]
    del data["adjacency"]
    del data["edge_features"]
    del data["y"]

    kwargs = {key : torch.tensor(data, dtype=torch.float64) for key, data in data.items()}
    kwargs["gen_index"] = kwargs["gen_index"].to(torch.int64)
    return Data(x=x, edge_index = edge_index, edge_attr = edge_weights, y = y, **kwargs)


def npz_to_torch(data, fixed_data):
    """
    Converts OPFSampler.jl info to pytorch data
    """
    x, y = data
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64).view(-1,1)

    edge_index = torch.tensor(fixed_data["adjacency"], dtype=torch.long)
    edge_weights = torch.tensor(fixed_data["edgeFeatures"], dtype=torch.float64)
    edge_index, edge_weights = to_undirected(edge_index, edge_attr=edge_weights)
    gen_index = torch.tensor(fixed_data["genIndex"], dtype=torch.int64)

    return Data(x=x, edge_index = edge_index, edge_attr = edge_weights, y = y, gen_index=gen_index)


def adjacency_to_torch(adj):
    """
    Converts standard ajacency matrix to torch.Tensor in format
    necessary for torch_geometric.data.Data.
    """
    row, col = np.where(adj != 0)
    return torch.Tensor(row), torch.Tensor(col)


def gnn_model_summary(model):
    "Gives a summary of model parameters"
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

#def gnn_summary_latex(model):
    #"""
    #Creates a summary of the model parameters which is exported to latex
#
    #Input:
        #model () : Model whose parameters we want to save
        #path (string) : Location where the latex file will be saved
    #"""
    #model_params_list = list(model.named_parameters())
    #layer, shape, num_of_params = ["Layer.Parameter"], ["Param Tensor Shape"], ["Number of Params"]
#
    #for elem in model_params_list:
        #layer.append(elem[0])
        #shape.append(list(elem[1].size()))
        #num_of_params.append(torch.tensor(elem[1].size()).prod().item())
    #total_data = zip(layer, shape, num_of_params)
#
    #table = tabulate(total_data, headers= "firstrow", tablefmt='latex')[:-14]
#
#
    #total_params = sum([param.nelement() for param in model.parameters()])
    #total_entry = f"\n Total Parameters & {total_params}"
    #num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #trainable_entry = f"\\\\\n Trainable Parameters & {num_trainable_params}"
    #untrainable_entry = f"\\\\\n Untrainbale Parameters & {total_params - num_trainable_params}"
    #end = "\n\\end{tabular}"
    #return table + total_entry + trainable_entry + untrainable_entry + end

def save_to_latex(latex_code, file):
    with open(file, "w") as out:
        out.write(latex_code)

def gnn_to_latex(model, file):
    latex_code = gnn_summary_latex(model)
    save_to_latex(latex_code, file)

def cfg_to_latex(cfg, file):
    "Creates table with training configuration"
    #Removing unnecessary information
    ignore = ["path", "test", "momentum", "betas", "lr", "model", "eval"]
    cfg = [[key,val] for key,val in cfg.items() if all([word not in key for word in ignore])]
    cfg.sort()
    table = tabulate(cfg, tablefmt='latex')
    save_to_latex(table, file)



def plot_loss(train_loss, test_loss, save = False):

    if save:
        mlp.use("pgf")
        mlp.rcParams.update({"pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,})
        fig, ax = plt.subplots(1, 1, figsize=set_size(483.69687, fraction=.5))
    else:
        fig, ax = plt.subplots(1,1)

    train, = ax.plot(train_loss, label="Train Loss")
    test, = ax.plot(test_loss, label="Test Loss")
    ax.legend(handles=[train, test])
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")

    if save:
        fig.savefig("loss.pgf")
    else:
        plt.show()

def plot_R2(values, save = False):

    if save:
        mlp.use("pgf")
        mlp.rcParams.update({"pgf.texsystem": "pdflatex",
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': True,})
        fig, ax = plt.subplots(1, 1, figsize=set_size(483.69687, fraction=.5))
    else:
        fig, ax = plt.subplots(1,1)

    R2, = ax.plot(values, label="R2")
    ax.legend(handles=[R2])
    plt.xlabel("Epochs")
    plt.ylabel("R2")

    if save:
        fig.savefig("R2.pgf")
    else:
        plt.show()

def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).

    **Graph-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, ImbalancedSampler

        sampler = ImbalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset or Data): The dataset from which to sample the data,
            either given as a :class:`~torch_geometric.data.Dataset` or
            :class:`~torch_geometric.data.Data` object.
        input_nodes (Tensor, optional): The indices of nodes that are used by
            the corresponding loader, *e.g.*, by
            :class:`~torch_geometric.loader.NeighborLoader`.
            If set to :obj:`None`, all nodes will be considered.
            This argument should only be set for node-level loaders and does
            not have any effect when operating on a set of graphs as given by
            :class:`~torch_geometric.data.Dataset`. (default: :obj:`None`)
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to :obj:`None`, will sample as much elements as there
            exists in the underlying data. (default: :obj:`None`)
    """
    def __init__(
        self,
        dataset: Union[Data, Dataset, List[Data]],
        input_nodes: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
    ):

        if isinstance(dataset, Data):
            y = dataset.y.view(-1)
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.data.y.view(-1)
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1. / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)

def discrete_loss(output, target):
    """
    DEPRECATED
    Used to calculate a loss based on 10 class classification


    """
    output_=output.clone()
    labels_=target.clone()
    discrete_array = torch.eq(torch.floor(output_*10), torch.floor(labels_*10))
    loss = torch.sum(discrete_array*-1+1).float()
    loss.requires_grad = True
    return loss/len(output)

def count_missclassified(output,target):
    """
    DEPRECATED
    Counts the missclassified instances

    Parameters
    ----------
    output : torch.tensor
        output tensor
    target : torch.tensor
        target tensor

    Returns
    -------
    count : int
        number of missclassified instances

    """
    count = 0
    missclassified = np.zeros(2000)
    for i in range(len(output)):
        if output[i] > target[i]* 0.8 and output[i] < target[i]*1.2:
            continue
        elif target[i] == 0 and output[i] < 0.01:
            continue
        else:
            count +=1
            missclassified[i] = output[i]
    plt.bar(range(2000),missclassified)
    return count

def get_zero_buses():
    """
    DEPRECATED

    Returns
    -------
    all_instances_zero : TYPE
        DESCRIPTION.

    """
    path_from_processed='processed/'#'/p/tmp/tobiasoh/machine_learning/Ike_ajusted_nonans/processed/'

    #if all_instances_zero[i] == 0 it means that the nodelabel at node[i] is zero in all instances
    all_instances_zero = torch.zeros(2000)
    N_allzero = 2000
    for file in os.listdir(path_from_processed):
        if file.startswith('data'):
            data = torch.load(path_from_processed+file)

            for i in range(2000):
                if all_instances_zero[i] != 0:
                    continue
                elif data.node_labels[i] != 0:
                    all_instances_zero[i] == 1
                    N_allzero -= 1
                if not any(all_instances_zero == 0):
                    break
    print(f'Number of Nodes that have always zero loadshed: {N_allzero}')
    return all_instances_zero

def setup_searchspace(cfg):
    """
    Sets up the searchspace for a study based on the configuration file

    Parameters
    ----------
    cfg : preloaded json configuration file


    Returns
    -------
    search_space :dictrionary
        dictionary of the search space used by ray

    """

    search_space = {}
    #General Architecture
    if cfg['study::lr::lower'] != cfg['study::lr::upper']:
        search_space['LR'] = tune.uniform(cfg['study::lr::lower'], cfg['study::lr::upper'])
    if cfg['study::weight_decay_upper'] != cfg['study::weight_decay_lower']:
        search_space['weight_decay'] = tune.uniform(cfg['study::weight_decay_lower'], cfg['study::weight_decay_upper'])

    if cfg["study::layers_lower"] != cfg["study::layers_upper"]:
        search_space['num_layers'] = tune.quniform(cfg["study::layers_lower"], cfg["study::layers_upper"]+1, 1)
    if cfg["study::hidden_features_lower"] != cfg["study::hidden_features_upper"]:
        search_space['hidden_size'] = tune.loguniform(cfg["study::hidden_features_lower"], cfg["study::hidden_features_upper"]+1)
    if cfg["study::dropout_lower"] != cfg["study::dropout_upper"]:
        search_space['dropout'] = tune.quniform(cfg["study::dropout_lower"], cfg["study::dropout_upper"], 0.01)
    if cfg['study::skipcon']:
        search_space['use_skipcon'] = tune.uniform(0, 2)
    if cfg['study::batchnorm']:
        search_space['use_batchnorm'] = tune.uniform(0, 2)

    #Regression Head
    if cfg['study::reghead_size_lower'] != cfg['study::reghead_size_upper']:
        search_space['reghead_size'] = tune.loguniform(cfg['study::reghead_size_lower'], cfg['study::reghead_size_upper']+1)
    if cfg["study::reghead_layers_lower"] != cfg['study::reghead_layers_upper']:
        search_space['reghead_layers'] = tune.uniform(cfg["study::reghead_layers_lower"], cfg['study::reghead_layers_upper']+1)

    #Training
    if cfg['study::gradclip_lower'] != cfg['study::gradclip_upper']:
        search_space['gradclip'] = tune.uniform(cfg['study::gradclip_lower'], cfg['study::gradclip_upper'])
    if cfg['study::masking']:
        search_space['use_masking'] = tune.uniform(0, 2)
        if cfg['study::mask_bias_lower'] != cfg['study::mask_bias_upper']:
            search_space['mask_bias'] = tune.quniform(cfg['study::mask_bias_lower'], cfg['study::mask_bias_upper'], 0.1)
    if cfg['study::loss_type']:
        search_space['loss_type'] = tune.uniform(0,2)
        if cfg['study::loss_weight_lower'] != cfg['study::loss_weight_upper']:
            search_space['loss_weight'] = tune.loguniform(cfg['study::loss_weight_lower'], cfg['study::loss_weight_upper'])

    #TAG configuration
    if cfg['study::tag_jumps_lower'] != cfg['study::tag_jumps_upper']:
        search_space['K'] = tune.uniform(cfg['study::tag_jumps_lower'], cfg['study::tag_jumps_upper']+1)

    #GAT and GraphTransformer configuration
    if cfg["study::heads_lower"] != cfg["study::heads_upper"]:
        search_space['heads'] = tune.uniform(cfg["study::heads_lower"], cfg["study::heads_upper"]+1)
    if cfg['study::gat_dropout_lower'] != cfg['study::gat_dropout_upper']:
        search_space['gat_dropout'] = tune.uniform(cfg['study::gat_dropout_lower'], cfg['study::gat_dropout_upper'])

    #Node2Vec configuration
    if cfg['study::embedding_dim_lower'] != cfg['study::embedding_dim_upper']:
        search_space['embedding_dim'] = tune.uniform(cfg['study::embedding_dim_lower'], cfg['study::embedding_dim_upper']+1)
    if cfg['study::walk_length_lower'] != cfg['study::walk_length_upper']:
        search_space['walk_length'] = tune.uniform(cfg['study::walk_length_lower'], cfg['study::walk_length_upper']+1)
    if cfg['study::context_size_lower'] != cfg['study::context_size_upper']:
        search_space['context_size'] = tune.uniform(cfg['study::context_size_lower'], cfg['study::context_size_upper']+1)
    if cfg['study::walks_per_node_lower'] != cfg['study::walks_per_node_upper']:
        search_space['walks_per_node'] = tune.uniform(cfg['study::walks_per_node_lower'], cfg['study::walks_per_node_upper']+1)
    if cfg['study::num_negative_samples_lower'] != cfg['study::num_negative_samples_upper']:
        search_space['num_negative_samples_lower'] = tune.uniform(cfg['study::num_negative_samples_lower'], cfg['study::num_negative_samples_upper']+1)
    if cfg['study::p_lower'] != cfg['study::p_upper']:
        search_space['p'] = tune.loguniform(cfg['study::p_lower'], cfg['study::p_upper'])
    if cfg['study::q_lower'] != cfg['study::q_upper']:
        search_space['q'] = tune.loguniform(cfg['study::q_lower'], cfg['study::q_upper'])

    return search_space

def setup_params_from_search_space(search_space, params):
    """
    params must already initiated by setup_params which will put the regular values from the cfg file
    setup_params_from_config then overrides the studied values with values from the search_space

    Parameters
    ----------
    search_space : the search_space created by setup_searchspace
    params : the parameters setup by setup_params

    Returns:
    -------
    params

    """
    updated_params = params
    print('Setup params from search space')
    for key in search_space.keys():
        if key in ['LR', 'weight_decay']:
            updated_params[key] = 10**search_space[key]
        else:
            updated_params[key] = search_space[key]
    return updated_params

def setup_params(cfg, mask_probs, num_features, num_edge_features):
    """
    Sets up the parameters dictionary for building and training a model

    Parameters
    ----------
    cfg : preloaded json configuration file

    mask_probs : float array
        probabilities for node masking
    num_features : int
        number of node features in the data
    num_edge_features : int
        number of edge features in the data

    Returns
    -------
    params : dict
        parameter dictionary

    """

    params = {
        'LR' :  cfg['optim::LR'],
        'weight_decay'   :   cfg['optim::weight_decay'],

        "num_features"          :   num_features,
        "num_edge_features"     :   num_edge_features,
        "num_targets"           :   1,

        "num_layers"    :   cfg['num_layers'],
        "hidden_size"   :   cfg['hidden_size'],

        "reghead_size"  :   cfg['reghead_size'],
        "reghead_layers":   cfg['reghead_layers'],

        "dropout"       :   cfg["dropout"],

        "use_batchnorm" :   cfg['use_batchnorm'],
        "gradclip"      :   cfg['gradclip'],
        "use_skipcon"   :   cfg['use_skipcon'],
        "use_masking"   :   cfg['use_masking'],
        'mask_bias'     :   cfg['mask_bias'],
        "mask_probs"    :   mask_probs,
        "loss_weight"   :   cfg['weighted_loss_factor'],

        #Params for GAT and GraphTransformer
        "heads"         :   cfg['num_heads'],
        'gat_dropout'   :   cfg['gat_dropout'],

        #Params for TAG
        "K"     :   cfg['tag_jumps'],

        #Params for LSTM
        "num_conv_targets"  :   cfg['num_conv_targets'],
        'lstm_hidden_size'  :   cfg['lstm_hidden_size'],
        'num_lstm_layers'   :   cfg['num_lstm_layers'],
        'len_sequence'      :   12,

        #Params for Node2vec
        'embedding_dim'   :   cfg['embedding_dim'],
        'walk_length'     :   cfg['walk_length'],
        'context_size'    :   cfg['context_size'],
        'walks_per_node'  :   cfg['walks_per_node'],
        'num_negative_samples'    :   cfg['num_negative_samples'],
        'p'     :   cfg['p'],
        'q'     :   cfg['q'],
    }
    return params

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



class weighted_loss_label(torch.nn.Module):
    """
    weights the loss with a constant factor depending on wether the label is >0 or not
    """
    def __init__(self, factor):
        super(weighted_loss_label, self).__init__()
        self.factor = torch.sqrt(factor)
        self.baseloss= torch.nn.MSELoss(reduction='mean')

    def forward(self, output, labels):
        print('Using weighted loss label')
        output_ = output.clone()
        labels_ = labels.clone()
        output_[labels>0] = output_[labels>0]*self.factor
        labels_[labels>0] = labels_[labels>0].clone()*self.factor
        return self.baseloss(self.factor*output_,self.factor*labels_)


class weighted_loss_var(torch.nn.Module):
    """
    weights the loss depending on the label variance at each node
    """
    def __init__(self, var, device):
        super(weighted_loss_var, self).__init__()
        self.weights = torch.sqrt(var).to(device)
        self.baseloss = torch.nn.MSELoss(reduction='mean').to(device)

    def forward(self, output ,labels):
        output_ = output.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        labels_ = labels.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        return self.baseloss(output_.reshape(-1), labels_.reshape(-1))
