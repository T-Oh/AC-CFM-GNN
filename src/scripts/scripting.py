import torch
import numpy as np

static = torch.load('processed/data_static.pt')
a = torch.load('processed/scenario_1111/data_1111_2.pt')
print(a)
inact = torch.nonzero(a.edge_labels == 0)
inactive_edge_indices = np.array(static.edge_index[:, inact])[:,:,0]
inactive_edge_indices = np.sort(inactive_edge_indices, axis=0)
print(inactive_edge_indices.shape)

inactive_edge_indices = inactive_edge_indices[:,np.argsort(inactive_edge_indices[0,:])]
for i in range(len(inactive_edge_indices[0])):
    print(inactive_edge_indices[0,i], inactive_edge_indices[1,i])

