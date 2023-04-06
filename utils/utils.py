from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch.serialization import save
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np
import matplotlib.pyplot as plt

#from tabulate import tabulate

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
        fig.savefig("/home/jan/pik/report/loss.pgf")
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
        fig.savefig("/home/jan/pik/report/R2.pgf")
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
    output_=output.clone()
    labels_=target.clone()
    discrete_array = torch.eq(torch.floor(output_*10), torch.floor(labels_*10))
    loss = torch.sum(discrete_array*-1+1).float()
    loss.requires_grad = True
    return loss/len(output)

def count_missclassified(output,target):
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

def weighted_loss_by_label(output, label, factor=10):
    loss = 0
    for i in range(len(output)):
       if label[i] > 0:
           loss += (output[i]-label[i])**2*factor
       else: 
           loss += (output[i]-label[i])**2
    return loss/2000
           
    