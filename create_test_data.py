import torch
from torch_geometric.data import Data
import os
node_features_a = torch.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
node_features_b = torch.tensor([[7,7,7,7,7,7],[8,8,8,8,8,8],[9,9,9,9,9,9]])
adj = torch.tensor([[0,1,1,2],[1,0,2,1]])
edge_attr_a = torch.tensor([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
edge_attr_b = torch.tensor([[8,8,8,8,8,8,8],[9,9,9,9,9,9,9]])
print(node_features_a.shape)
print(adj.shape)
data = Data(x=torch.cat((node_features_a,node_features_a+10,node_features_a+100)), edge_index=torch.cat((adj, adj, adj)), edge_attr=torch.cat((edge_attr_a, edge_attr_a+10, edge_attr_a+100)), y=1)
torch.save(data, os.path.join('processed/', f'data_{1}_{1}.pt'))

data = Data(x=torch.cat((node_features_b,node_features_b+10)), edge_index=torch.cat((adj, adj)), edge_attr=torch.cat((edge_attr_b, edge_attr_b+10)), y=2)
torch.save(data, os.path.join('processed/', f'data_{2}_{2}.pt'))