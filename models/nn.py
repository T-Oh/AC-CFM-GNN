from torch_geometric.nn import Sequential, GCNConv, global_add_pool, global_mean_pool
from torch.nn import Module, Sigmoid, ReLU, Dropout, Linear


class NN(Module):
    """Graph convolutional model with pooling layer for graph classification"""

    def __init__(self, num_features=1, num_targets = 1, hidden_size=16, num_layers=1, dropout=.0):
        super().__init__()
        self.hid1=Linear(num_features,)

        for i in range(num_layers):
            if not layers:
                layers.append((GCNConv(num_node_features, hidden_size), 'x, edge_index, edge_weight -> x1'))
            else:
                layers.append((Sigmoid(), f'x{i} -> x{i}a'))
                layers.append((Dropout(p=dropout), f'x{i}a -> x{i}d'))
                layers.append((GCNConv(hidden_size, hidden_size), f'x{i}d, edge_index, edge_weight -> x{i+1}'))



        if pool == "add":
            layers.append((global_add_pool, f'x{num_layers}, batch -> y'))
        elif pool == "mean":
            layers.append((global_mean_pool, f'x{num_layers}, batch -> y'))
        else:
            raise ValueError("pool must be either mean or add")

        layers.append((Dropout(p=dropout), 'y -> yd'))
        layers.append((Linear(hidden_size, 1), 'yd -> yl'))

        self.model = Sequential('x, batch, edge_index, edge_weight', layers)
        self.model.to(float)

    def forward(self, data):
        x, batch, edge_index, edge_weight  = data.x, data.batch, data.edge_index, data.edge_attr
        return self.model(x, batch, edge_index, edge_weight)