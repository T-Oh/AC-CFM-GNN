from torch_geometric.nn import GATConv, GATv2Conv, BatchNorm
from torch.nn import Module, Dropout, Linear, LeakyReLU, ModuleList, Sigmoid
import numpy as np
import torch
#from utils.utils import dirichlet_energy, symmetrically_normalized_laplacian
import networkx as nx
import matplotlib.pyplot as plt
import random


class GATpaper(Module):
    """
    Graph Transformer from paper:
    https://ieeexplore.ieee.org/document/9881910
    In the original paper they use GATConv with static attention
    I switched it to GATv2Conv with dynamic attention, which is more expressive
    """

    def __init__(self, num_node_features=6, num_edge_features=2, num_targets=2,
                 hidden_size=50, num_layers=2, reghead_size=500, reghead_layers=1,
                 dropout=0.0, gat_dropout=0.0, num_heads=1, use_skipcon=False,
                 use_batchnorm=False):
        """
        INPUT
        num_node_features   :   int
            number of node features in data
        num_edge_features   :   int
            number of edge features in the data
        num_targets         :   int
            number of labels in the data
        hidden_size         :   int
            the number of hidden features to be used
        num_layers          :   int
            the number of layers to be used
        reghead_size        :   int
            number of hidden features of the regression head
        reghead_layers      :   itn
            number of regression head layers
        dropout             :   float
             the dropout to be applied
        gat_dropout        :   float
             dropout applied within the GATv2Conv layers
        num_heads          :   int
             number of attention heads
        use_batchnorm       :   bool
             whether batchnorm should be applied
        use_skipcon        :   boo
             whether skip connections should be applied

        """

        super(GATpaper, self).__init__()

        # Params
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.reghead_size = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.num_heads = int(num_heads)
        self.use_skipcon = bool(int(use_skipcon))
        self.use_batchnorm = bool(int(use_batchnorm))

        # Conv Layers
        self.conv = GATConv(num_node_features, num_node_features, edge_dim=2, dropout=gat_dropout, concat=False,
                            add_self_loops=False, heads=self.num_heads)
        self.convLayers = ModuleList([GATConv(num_node_features, num_node_features, edge_dim=2, dropout=gat_dropout,
                                              concat=False, add_self_loops=False, heads=self.num_heads) for i in
                                      range(self.num_layers)])

        # Regression Head Layers
        self.reshape = Linear(num_node_features, self.reghead_size)
        self.regHead = Linear(self.reghead_size, self.reghead_size)
        self.regHeadLayers = ModuleList(
            Linear(self.reghead_size, self.reghead_size) for i in range(self.reghead_layers - 1))
        self.endLinear1 = Linear(self.reghead_size, 1, bias=True)
        self.endLinear2 = Linear(self.reghead_size, 1, bias=True)

        # Additional Layers
        self.relu = LeakyReLU()
        self.sigmoid = Sigmoid()
        self.linear_ff1 = ModuleList(Linear(num_node_features, num_node_features) for i in range(self.num_layers))
        self.linear_ff2 = ModuleList(Linear(num_node_features, num_node_features) for i in range(self.num_layers))

        self.batchnorm = BatchNorm(num_node_features, track_running_stats=True)

    def forward(self, data, output_dirichlet_energy=False):
        x = data.x

        num_nodes = x.shape[0]

        edge_index = data.edge_index % num_nodes
        edge_attr = data.edge_attr
        # print(f'{x.size()=}')
        node_features = x.view(-1, num_nodes, 6)
        batchsize = node_features.shape[0]

        device = x.device
        # x = x.float()
        # edge_index = edge_index.float()
        edge_attr = edge_attr.float()
        # if output_dirichlet_energy:
        #     snl = symmetrically_normalized_laplacian(edge_index)
        #     print(f'{snl.size()=}')
        #     dir_energy = []
        #     print(f'start empty {dir_energy=}')
        #     temp_dir_energy = dirichlet_energy(snl, x, batchsize)
        #     dir_energy.append(temp_dir_energy)

        x2 = x
        for i in range(self.num_layers):  # for it to work we need num_layers >= 2 as in the paper
            # GAT
            print('layer', i)
            # print(f'{edge_index.size()=}')

            # print(f'in GAT{edge_attr.size()=}')
            # print('x, edge_index, edge_attr', x.size(), edge_index.size(), edge_attr.size())
            x, (edge_index_att, attention_weights) = self.convLayers[i](x2, edge_index=edge_index, edge_attr=edge_attr,
                                                                        return_attention_weights=True)

            # if output_dirichlet_energy:
            #     print(f'{x.size()=}')
            #     temp_dir_energy = dirichlet_energy(snl, x, batchsize)
            #     dir_energy.append(temp_dir_energy)

            # if torch.allclose(edge_index, edge_index_att):
            #     print('edge index is ok')
            # else:
            #     print('edge index wrong')
            # print('edge index size', edge_index.size())
            # print('attention weights size', attention_weights[:,0].size())
            # print('attention_weights', attention_weights)

            # # Create a graph
            # G = nx.Graph()

            # # Add edges with corresponding attention weights
            # for edge, weight in zip(edge_index, attention_weights[:,0].detach().numpy()):
            #     G.add_edge(edge[0], edge[1], weight=weight)

            # # # Extract attention weights
            # # weights = [data['weight'] for _, _, data in G.edges(data=True)]

            # # Draw the graph
            # pos = nx.spring_layout(G)  # You can use other layout algorithms too
            # nx.draw(G, pos, with_labels=True, edge_color=attention_weights[:,0].detach().numpy(), edge_cmap=plt.cm.Blues, width=2, alpha=0.7)
            # plt.colorbar()  # Add a colorbar to show the correspondence between colors and weights
            # plt.show()

            # G = nx.Graph()

            # # Add edges with corresponding attention weights
            # for i in range(edge_index.size(1)):
            #     src, tgt = edge_index[:, i].tolist()
            #     weight = attention_weights[i,0].item()
            #     G.add_edge(src, tgt, weight=weight)

            # Extract attention weights
            # weights = [weight for _, _, data in G.edges(data=True)]

            # Draw the graph
            # pos = nx.spring_layout(G)  # You can use other layout algorithms too
            # nx.draw(G, pos, with_labels=False, node_size=10, edge_color=weights, edge_cmap=plt.cm.Blues, width=0.5, alpha=0.7)
            # plt.colorbar(label='Attention Weight')  # Add a colorbar to show the correspondence between colors and weights
            # plt.show()

            # G = nx.gnp_random_graph(10,0.3)
            # for u,v,d in G.edges(data=True):
            #     d['weight'] = random.random()

            # edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())

            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)
            # plt.savefig('edges0.png')
            # edge_index_att = edge_index_att[:, :-2000]
            # attention_weights = attention_weights[:-2000, :]
            # G = nx.Graph()
            # for i in range(edge_index_att.size(1)):
            #     src, tgt = edge_index_att[:, i].tolist()
            #     weight = attention_weights[i,0].item()
            #     G.add_edge(src, tgt, weight=weight)

            # edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())

            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, node_size = 1, width=1, edge_cmap=plt.cm.Blues)
            # plt.savefig('edges_GATv2.png')

            # for param in self.convLayers[i].parameters():
            #     print('conv params', param)
            #     print('conv gradients', param.grad)

            # x.retain_grad()
            # print('x grad after conv', x.grad)
            x = self.relu(x)  # originally it was sigmoid here

            # print('x', x)
            # Add & Norm
            if i == 0:
                x = x + data.x
            elif i == 1:
                x = x + data.x + x1 + x2
            else:
                x = x + x2 + x3
            x1 = (x - torch.mean(x)) / torch.std(x)

            # Feed-forward
            x = self.linear_ff1[i](x1)
            x = self.relu(x)
            x = self.linear_ff2[i](x)

            # for param in self.linear_ff1.parameters():
            #     print('linear ff1 params', param)
            #     print('linear ff1 gradients', param.grad)
            # Add & Norm
            if i == 0:
                x = x + x1 + data.x
            elif i == 1:
                x = x + x1 + x2 + x3 + data.x
            else:
                x = x + x1 + x2 + x3
            x3 = x1

            x2 = (x - torch.mean(x)) / torch.std(x)

        # if output_dirichlet_energy:
        #     print('dirichlet energy', dir_energy)
        '''
        #Plotting Dirichlet energy
        plt.plot(torch.tensor(dir_energy, device = 'cpu'), label='Dirichlet energy')
        plt.legend()
        plt.title('Dirichlet energy')
        plt.savefig('Dirichlet_energy.png')
        '''

        # Regression Head
        x = self.reshape(x)
        for i in range(self.reghead_layers - 1):
            x = self.regHeadLayers[i](x)
            x = self.relu(x)
        Vpred_real = self.endLinear1(x)
        Vpred_imag = self.endLinear2(x)

        outputs = torch.cat((Vpred_real, Vpred_imag), dim=1)
        # print('gtrans2.py outputs', outputs)
        # if output_dirichlet_energy:
        #     return outputs, dir_energy
        # else:
        return outputs
