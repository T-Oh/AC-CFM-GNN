from torch_geometric.nn import Sequential, GATv2Conv, global_add_pool, global_mean_pool, RGATConv
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Tanh
import torch



class TAGNet01(Module):
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=128, num_layers=3, dropout=.15):
        super(TAGNet01, self).__init__()
        self.conv1 = TAGConv(num_node_features, hidden_size,bias=False,K=0).to(float)
        self.conv2 = TAGConv(hidden_size, hidden_size,bias=False,K=0).to(float)
        #self.endconv = TAGConv(hidden_size, num_targets,bias=False).to(float)
        self.endLinear = Linear(hidden_size,num_targets,bias=False).to(float)
        self.endSigmoid = Sigmoid()
        self.endTanh=Tanh()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.num_layers = num_layers

    def forward(self, data):
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index.type(torch.int64), data.edge_attr.float()


        PRINT=True
        
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index)
            print(edge_weight)
            
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        print('XSHAPE AFTER CONV1')
        print(x.shape)

        for _ in range(self.num_layers - 2):
            x = self.relu(x)
            #x = self.endSigmoid(x)
            print(x)
            x = self.dropout(x)
            print(x)
            x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
            print(x)

        x = self.relu(x)
        #print(x)
        x = self.dropout(x)
        #print(x)

        #x = self.endconv(x=x, edge_index=edge_index, edge_weight=edge_weight)   #produces negative values for some reason
        #print(x)
        x=self.endLinear(x)
        #print("Linear output")
        #print(x)
        x = self.pool(x, batch)
        
        #print("Pool")
        #print(x)
        x = self.endSigmoid(x)
        #print(x)
        print("END")
        return x


class RGAT(Module):
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=16, num_layers=1, dropout=0.05):
        super(SAGE, self).__init__()
        self.num_layers=num_layers
        #self.conv1=GATv2Conv(num_node_features,hidden_size,edge_dim=2,heads=1).to(float)
        self.conv1=SAGEConv(num_node_features,hidden_size).to(float)
        self.conv2=SAGEConv(hidden_size,hidden_size).to(float)
        
        self.relu = ReLU()
        self.endLinear = Linear(hidden_size,num_targets,bias=True).to(float)
        self.endSigmoid=Sigmoid()
        self.pool = global_mean_pool    #global add pool does not work for it produces too large negative numbers
        self.dropout = Dropout(p=dropout)
        
    def forward(self, data):
        #x, edge_index, edge_weight = data['bus'].x, data['bus','bus'].edge_index.type(torch.int64), data['bus','bus'].edge_attr
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index.type(torch.int64), data.edge_attr.float()


        PRINT=False
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index)
            print(edge_weight)
            
        #x = self.convsingle(x=x, edge_index=edge_index, edge_weight=edge_weight)
        #x=self.conv1(x=x, edge_index=edge_index,edge_attr=edge_weight)
        x=self.conv1(x=x, edge_index=edge_index)
        
        
        for _ in range(self.num_layers - 1):
            x = self.relu(x)
            #print(x)
            #x = self.dropout(x)
            #print(x)
            x = self.conv2(x=x, edge_index=edge_index)
            #print(x)
        
        x = self.relu(x)
        #print(x)
        x = self.dropout(x)
        #print(x)
        x=self.endLinear(x)
        #print(f'SHAPE AFTER ENDLINEAR {x.shape}')
        #print("Pool")
        #print(x)
        x = self.pool(x,batch)
        x = self.endSigmoid(x)
        x.type(torch.FloatTensor)
        #print(x)
        #print("END")
        return x

