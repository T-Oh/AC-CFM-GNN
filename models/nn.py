from torch_geometric.nn import Sequential, GCNConv, global_add_pool, global_mean_pool, Node2Vec
from torch.nn import Module, Sigmoid, ReLU, Dropout, Linear



class baseline(Module):
    """Graph convolutional model with pooling layer for graph classification"""

    def __init__(self, edge_index, embedding_dim=128, walk_length=10, context_size=1, walks_per_node=1):
        super().__init__()
        self.node2vec = Node2Vec(edge_index=edge_index, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node)
        self.linear = Linear(in_features=embedding_dim*2000, out_features=2000)


    def forward(self, data):
        x, batch, edge_index, edge_weight  = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.node2vec(batch)
        print(x.shape)
        x = self.linear(x.reshape(-1))
        print(x)
        
        return x