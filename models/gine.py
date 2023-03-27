from torch_geometric.nn import Sequential, GATv2Conv, global_add_pool, global_mean_pool, SAGEConv, GATConv, BatchNorm, GINEConv
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Tanh, BatchNorm1d, LeakyReLU
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
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reg_head_size=500, dropout=0.0, dropout_temp=1.0, num_heads=1, use_batchnorm = True, use_skipcon=False):
        super(GINE, self).__init__()
        #Params
        self.num_layers=num_layers
        self.use_skipcon = use_skipcon
        self.dropout_temp = dropout_temp
        
        #ConvLayers
        
        self.convLayer1 = GINEConv(
            nn.Sequential(
                Linear(num_node_features, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size),
                LeakyReLU()
                ), edge_dim=num_edge_features)
        
        self.convLayer2 = GINEConv(                 #Layers 2 to 5 are the same but batchnorm layers can not be reuse bcs of track_stats=True
            nn.Sequential(
                Linear(hidden_size, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size), 
                LeakyReLU()
                ), edge_dim=num_edge_features)
        self.convLayer3 = GINEConv(
            nn.Sequential(
                Linear(hidden_size, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size), 
                LeakyReLU()
                ), edge_dim=num_edge_features)
        self.convLayer4 = GINEConv(
            nn.Sequential(
                Linear(hidden_size, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size), 
                LeakyReLU()
                ), edge_dim=num_edge_features)
        self.convLayer5 = GINEConv(
            nn.Sequential(
                Linear(hidden_size, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size), 
                LeakyReLU()
                ), edge_dim=num_edge_features)

        
        #Additional Layers
        self.relu = LeakyReLU()
        self.endLinear = Linear(reg_head_size,num_targets,bias=True)
        self.endSigmoid=Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)
        self.regHead = Linear(hidden_size, reg_head_size)
        self.batchnorm = BatchNorm(hidden_size*num_heads,track_running_stats=False)

    def forward(self, data):
        
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index, data.edge_attr.float()

        PRINT=False
        if PRINT:
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)

        out = self.convLayer1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        #print(out)

    
        for i in range(self.num_layers - 1):
            if self.use_skipcon:
                if i==0:
                    skip_in = out.clone()
                    out = self.convLayer2(x=out, edge_index=edge_index,edge_attr = edge_weight)
                    regular_in = out.clone()
                if i==1:
                    out = self.convLayer3(x=skip_in+regular_in/2, edge_index=edge_index,edge_attr = edge_weight)
                    skip_in = regular_in.clone()
                    regular_in = out.clone()
                if i==2:
                    out = self.convLayer4(x=skip_in+regular_in/2, edge_index=edge_index,edge_attr = edge_weight)
                    skip_in = regular_in.clone()
                    regular_in = out.clone()
                if i==3:
                    out = self.convLayer5(x=skip_in+regular_in/2, edge_index=edge_index,edge_attr = edge_weight)
                    skip_in = regular_in.clone()
                    regular_in = out.clone()

            else:
                if i== 0:
                    out = self.convLayer2(x=out, edge_index=edge_index,edge_attr = edge_weight)
                if i== 1:
                    out = self.convLayer3(x=out, edge_index=edge_index,edge_attr = edge_weight)
                if i== 2:
                    out = self.convLayer4(x=out, edge_index=edge_index,edge_attr = edge_weight)
                if i== 3:
                    out = self.convLayer5(x=out, edge_index=edge_index,edge_attr = edge_weight)
            
            #print(out)

        
        #out = self.relu(out)
        #print(x)
        self.dropout.p = self.dropout.p*self.dropout_temp
        print(f'Dropout {self.dropout.p}')
        out = self.dropout(out)
        #print(out)
        #Regression Head
        out = self.regHead(out)
        #print(out)
        out = self.relu(out)
        #print(out)
        out = self.endLinear(out)
        #print(out)
        
        out.type(torch.DoubleTensor)
        #print(x)
        #print("END")
        return out
