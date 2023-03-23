from torch_geometric.nn import Sequential, GATv2Conv, global_add_pool, global_mean_pool, SAGEConv, GATConv, BatchNorm
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Tanh, LeakyReLU
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

class GAT(Module):
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reg_head_size=500, dropout=0.0, num_heads=1, use_batchnorm = True):
        super(GAT, self).__init__()
        #Params
        self.num_layers=num_layers
        self.use_batchnorm = use_batchnorm
        
        #Conv Layers
        self.conv1=GATv2Conv(num_node_features,hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout, heads=num_heads).to(float)
        self.conv2=GATv2Conv(hidden_size*num_heads,hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout,heads=num_heads).to(float)

        #Additional Layers
        self.relu = LeakyReLU()
        self.regHead = Linear(hidden_size*num_heads, reg_head_size)
        self.endLinear = Linear(reg_head_size ,num_targets,bias=True)
        self.endSigmoid = Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)       
        self.batchnorm = BatchNorm(hidden_size*num_heads,track_running_stats=False)

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


        
        #print(f'Before:\n {x}')
        for _ in range(self.num_layers - 1):
            if self.use_batchnorm:
                print('USING BATCHNORM')
                x = self.batchnorm(x)
                print(f'After:\n{x}')
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
        x = self.regHead(x)
        x=self.endLinear(x)
        #print(f'SHAPE AFTER ENDLINEAR {x.shape}')
        #print("Pool")
        #print(x)
        #x = self.pool(x,batch)
        #x = self.endSigmoid(x)
        x.type(torch.DoubleTensor)
        #print(x)
        #print("END")
        return x
