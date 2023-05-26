from torch_geometric.nn import Sequential, GCNConv, global_add_pool, global_mean_pool, Node2Vec
from torch.nn import Module, Sigmoid, LeakyReLU, Dropout, Linear
import torch



class ridge(Module):
    """Simple ridge model to use as baseline"""

    def __init__(self, num_node_features, hidden_size=128):
        super().__init__()
        self.linear = Linear(in_features=2000*num_node_features, out_features=2000)


    def forward(self, data):
        x  = data.x
        
        x = self.linear(x.reshape(-1))
        #print(x)
        
        return x
    
class MLP(Module):
    def __init__(self, num_node_features, hidden_size, num_layers):
        super().__init__()
        assert num_layers <= 3, 'A maximum of 3 layers implemented for MLP'
        self.num_layers = num_layers
        self.lin_single = Linear(num_node_features*2000, 2000)
        self.lin_in = Linear(in_features=2000*num_node_features, out_features = hidden_size)
        self.lin_hidden = Linear(in_features=hidden_size, out_features = hidden_size)
        self.lin_end = Linear(in_features=hidden_size, out_features = 2000)
        self.ReLu = LeakyReLU()
        
        
        
    def forward(self, data):
        x = data.x
        #print(f'MLP input shape: {x.shape}')
        if self.num_layers == 1:
            x = self.lin_single(x.reshape(-1))
        elif self.num_layers > 1:
            x = self.lin_in(x.reshape(-1))
            x = self.ReLu(x)
            if self.num_layers > 2:
                x = self.lin_hidden(x)
                x = self.ReLu(x)
            x = self.lin_end(x)
        return x
        