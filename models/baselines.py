from torch_geometric.nn import Sequential, GCNConv, global_add_pool, global_mean_pool, Node2Vec
from torch.nn import Module, Sigmoid, ReLU, Dropout, Linear
import torch



class ridge(Module):
    """Simple ridge model to use as baseline"""

    def __init__(self, num_node_features, hidden_size=128):
        super().__init__()
        self.linear = Linear(in_features=2000*num_node_features, out_features=2000)


    def forward(self, data):
        x, batch, edge_index, edge_weight  = data.x, data.batch, data.edge_index, data.edge_attr
   
        x = self.linear(x.reshape(-1))
        #print(x)
        
        return x
    
class node2vec(Module):
    """Node2Vec Baseline (unfinished)"""

    def __init__(self, edge_index, embedding_dim=128, walk_length=10, context_size=1, walks_per_node=1):
        super().__init__()
        self.node2vec = Node2Vec(edge_index=edge_index, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node)
        self.linear = Linear(in_features=2000*embedding_dim, out_features=2000)


    def forward(self, data):
        x, batch, edge_index, edge_weight  = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.node2vec(batch)
        print(x[0:50,0])
        #for i in range(len(x)):    
        x = self.linear(x.reshape(-1))
        print(x)
        
        return x
    
class mean(Module):
    """Baseline which simply predicts the mean power outage at every node"""
    
    def __init__(self, means):
        super().__init__()
        self.means = means
        self.linear = Linear(in_features=1, out_features=1)
        
    def forward(self, data):
        return self.means