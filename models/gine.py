from torch_geometric.nn import Sequential, GATv2Conv, global_add_pool, global_mean_pool, SAGEConv, GATConv, BatchNorm, GINEConv
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Tanh
import torch.nn as nn
import torch


#TESTING SUITE
"""

conv1 = GATv2Conv(2,2, edge_dim = 3, add_self_loops=False, dropout = 0.1)
conv1.type(torch.DoubleTensor)
data = torch.load('../processed/data_3_3.pt')

x, (edge_index, attention_weights)=conv1(x=data.x.double(), edge_index=data.edge_index, edge_attr=data.edge_attr.double(), return_attention_weights = True)

count = 0
for i in range(len(x)):
    if torch.isnan(x[i]).any():
        print(f'Node: {i}')
        count +=1
        for j in range(len(data.edge_index[0,:])):
            #print(data.edge_index[0,j])
            #print(data.edge_index[1,j])
            if data.edge_index[0,j] == i or data.edge_index[1,j] == i:
                print(data.edge_index[:,j])
                print(data.edge_attr[j,:])
        if count >3: break
        
assert False

"""

class GINE(Module):
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, dropout=0.0, num_heads=1, batchnorm = True):
        super(GINE, self).__init__()
        self.num_layers=num_layers
        GINE_linear1 = nn.Sequential(Linear(num_node_features, hidden_size))
        GINE_linear2 = nn.Sequential(Linear(hidden_size, hidden_size))
        self.conv1=GINEConv(GINE_linear1, edge_dim=num_edge_features)#.to(float)
        self.conv2=GINEConv(GINE_linear2, edge_dim=num_edge_features)#.to(float)

        self.relu = ReLU()
        self.endLinear = Linear(hidden_size*num_heads,num_targets,bias=True)
        self.endSigmoid=Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)
        
        self.batchnorm = BatchNorm(hidden_size*num_heads)
        self.use_batchnorm = batchnorm
    def forward(self, data):
        
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index, data.edge_attr.float()


        PRINT=False
        if PRINT:
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)
            
        #x = self.convsingle(x=x, edge_index=edge_index, edge_weight=edge_weight)
        #x=self.conv1(x=x, edge_index=edge_index,edge_attr=edge_weight)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)


        
        
        for _ in range(self.num_layers - 1):
            if self.use_batchnorm:
                print('USING BATCHNORM')
                x = self.batchnorm(x)
            x = self.relu(x)
            #print(x)
            #x = self.dropout(x)
            #print(x)
            x = self.conv2(x=x, edge_index=edge_index,edge_attr = edge_weight)
            #print(x)
        
        x = self.relu(x)
        #print(x)
        #x = self.dropout(x)
        #print(x)
        x=self.endLinear(x)
        #print(f'SHAPE AFTER ENDLINEAR {x.shape}')
        #print("Pool")
        #print(x)
        #x = self.pool(x,batch)
        x = self.endSigmoid(x)
        x.type(torch.DoubleTensor)
        #print(x)
        #print("END")
        return x
