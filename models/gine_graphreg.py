from torch_geometric.nn import BatchNorm, GINEConv, global_mean_pool
from torch.nn import Module, Dropout, Linear, BatchNorm1d, LeakyReLU
import torch.nn as nn




class GINEGraphReg(Module):
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reghead_size=500, reghead_layers =2,
                 dropout=0.0, use_skipcon=False, use_batchnorm=True):
        super(GINEGraphReg, self).__init__()
        #use_batchnorm does not have an effect on GINE since the batchnorm is automatically used inside the layers
        print('\n\nGINE INIT OUTPUT\n ')
        #Params
        self.num_layers=int(num_layers)
        self.use_skipcon = bool(int(use_skipcon))
        self.reghead_layers = int(reghead_layers)
        hidden_size = int(hidden_size)
        reghead_size = int(reghead_size)
        
        
        #ConvLayers        
        self.convLayer1 = GINEConv(
            nn.Sequential(
                Linear(num_node_features, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(),
                Linear(hidden_size,hidden_size),
                LeakyReLU()
                ), edge_dim=num_edge_features)

        self.gines = nn.ModuleList([GINEConv(                 #Layers 2 to 5 are the same but batchnorm layers can not be reuse bcs of track_stats=True
            nn.Sequential(
                Linear(hidden_size, hidden_size),
                BatchNorm1d(hidden_size),
                LeakyReLU(), 
                Linear(hidden_size,hidden_size), 
                LeakyReLU()
                ), edge_dim=num_edge_features) for i in range(self.num_layers-1)])
        
        
       
        #Regression Head Layers
        self.regHead1 = Linear(hidden_size, reghead_size)
        self.singleLinear = Linear(hidden_size, num_targets)
        self.regHeadLayers = nn.ModuleList(Linear(reghead_size, reghead_size) for i in range(self.num_layers-2))
        self.endLinear = Linear(reghead_size,num_targets,bias=True)

        
        #Additional Layers
        self.relu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        print(f'Dropoutrate {dropout}')
        self.batchnorm = BatchNorm(hidden_size,track_running_stats=True)

    def forward(self, data):
        print('\n\nGINE FORWARD OUTPUT\n ')
        
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index, data.edge_attr.float()
        PRINT=False
        if PRINT:
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)
        
        x = self.convLayer1(x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.relu(x)
        x = self.dropout(x)


        #Arranging Conv Layers
        if self.use_skipcon:   
            print('Using Skipcon')
            for i in range(self.num_layers -1):
                x_ = self.gines[i](x, edge_index=edge_index, edge_attr=edge_weight)
                x = (self.relu(x_)+x)/2
                x = self.dropout(x)
        else:
            for i in range(self.num_layers-1):
                x = self.gines[i](x, edge_index=edge_index, edge_attr=edge_weight)
                x = self.relu(x)
                x = self.dropout(x)

        x = global_mean_pool(x, batch)
        #Regression Head
        if self.reghead_layers == 1:
                x = self.singleLinear(x)


        elif self.reghead_layers > 1:
            x = self.regHead1(x)

            x = self.relu(x)
            for i in range(self.reghead_layers-2):
                x = self.regHeadLayers[i](x)
                x = self.relu(x)
                #print(out)
            x = self.endLinear(x)
        
        #print(x)
        #print("END")
        return x
