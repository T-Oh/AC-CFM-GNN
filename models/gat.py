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
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reghead_size=500, reghead_layers=1, dropout=0.0, num_heads=1):
        super(GAT, self).__init__()
        
        assert num_layers <= 5, 'A maximum of 5 layers implemented for GAT'
        #Params
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.reghead_size = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.num_heads = int(num_heads)
        
        
        #Conv Layers
        self.conv1=GATv2Conv(num_node_features,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout, heads=self.num_heads).to(float)
        self.conv2=GATv2Conv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout,heads=self.num_heads).to(float)
        self.conv3=GATv2Conv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout,heads=self.num_heads).to(float)
        self.conv4=GATv2Conv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout,heads=self.num_heads).to(float)
        self.conv5=GATv2Conv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = dropout,heads=self.num_heads).to(float)
        

        #Additional Layers
        self.relu = LeakyReLU()
        self.regHead1 = Linear(self.hidden_size*self.num_heads, self.reghead_size)
        self.regHead2 = Linear(self.reghead_size, self.reghead_size)
        self.endLinear = Linear(self.reghead_size ,num_targets,bias=True)
        self.singleLinear = Linear(self.hidden_size*self.num_heads, num_targets)
        self.endSigmoid = Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)       
        self.batchnorm = BatchNorm(self.hidden_size*self.num_heads,track_running_stats=True)

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
        for layer in range(self.num_layers - 1):
            x = self.relu(x)
            #print(x)
            #x = self.dropout(x)
            #print(x)
            if layer == 0:
                x = self.conv2(x=x, edge_index=edge_index,edge_attr = edge_weight)
            elif layer == 1:
                x = self.conv3(x=x, edge_index=edge_index,edge_attr = edge_weight)
            elif layer == 2:
                x = self.conv4(x=x, edge_index=edge_index,edge_attr = edge_weight)
            elif layer == 3:
                x = self.conv5(x=x, edge_index=edge_index,edge_attr = edge_weight)
            #print(x)
            
        x = self.dropout(x)
        
        if self.reghead_layers == 1:
                x = self.singleLinear(x)
        
        
        elif self.reghead_layers > 1:
            x = self.regHead1(x)
        
            x = self.relu(x)
            for i in range(self.reghead_layers-2):
                x = self.regHead2(x)
        
                #print(out)
                x = self.relu(x)
                #print(out)
            x = self.endLinear(x)
        #print(f'SHAPE AFTER ENDLINEAR {x.shape}')
        #print("Pool")
        #print(x)
        #x = self.pool(x,batch)
        #x = self.endSigmoid(x)
        x.type(torch.DoubleTensor)
        #print(x)
        #print("END")
        return x
