from torch_geometric.nn import  GATv2Conv, BatchNorm, global_mean_pool
from torch.nn import Module, Dropout, Sigmoid, Linear, LeakyReLU, ModuleList


class GATGraphReg(Module):
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reghead_size=500, reghead_layers=1,
                 dropout=0.0, gat_dropout=0.0,  num_heads=1, use_skipcon=False, use_batchnorm=False):
        super(GATGraphReg, self).__init__()
        
        #Params
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.reghead_size = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.num_heads = int(num_heads)
        self.use_skipcon = bool(int(use_skipcon))
        self.use_batchnorm = bool(int(use_batchnorm))
        print(f'Dropout Rate {dropout}')
        print(f'Gat Dropout Rate {gat_dropout}')
        print(f'Num Heads {self.num_heads}')
        print(f'Num Layers {self.num_layers}')
        print(f'Hidden Size {self.hidden_size}')
        print(f'Reghead Size {self.reghead_size}')
        print(f'Reghead Layers {self.reghead_layers}')
        
        
        #Conv Layers
        self.conv1=GATv2Conv(num_node_features,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = gat_dropout, heads=self.num_heads)
        self.convLayers = ModuleList([GATv2Conv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, add_self_loops=True, dropout = gat_dropout, heads=self.num_heads) for i in range(self.num_layers-1)])
      
        #Regression Head Layers
        self.regHead1 = Linear(self.hidden_size*self.num_heads, self.reghead_size)
        self.singleLinear = Linear(self.hidden_size*self.num_heads, num_targets)
        self.regHeadLayers = ModuleList(Linear(self.reghead_size, self.reghead_size) for i in range(self.num_layers-2))
        self.endLinear = Linear(self.reghead_size,num_targets,bias=True)
        
        #Additional Layers
        self.relu = LeakyReLU()
        self.dropout = Dropout(p=dropout)       
        self.batchnorm = BatchNorm(self.hidden_size*self.num_heads,track_running_stats=True)


    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.float()


        PRINT=False
        if PRINT:
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)
        
        
        x = self.conv1(x, edge_index=edge_index, edge_attr=edge_weight)
        if self.use_batchnorm:
            x=self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)


        #Arranging Conv Layers
        for i in range(self.num_layers -1):
            if self.use_skipcon:   
                print('Using Skipcon')
                x_ = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                if self.use_batchnorm:
                    print('Using Batchnorm')
                    x_ = self.batchnorm(x_)
                x = (self.relu(x_)+x)/2
                x = self.dropout(x)
            else:
                x = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                if self.use_batchnorm:
                    print('Using Batchnorm')
                    x = self.batchnorm(x)
                x = self.relu(x)
                x = self.dropout(x)
                
        x = global_mean_pool(x, batch)
        
        if self.reghead_layers == 1:
                x = self.singleLinear(x)
        
        
        elif self.reghead_layers > 1:
            x = self.regHead1(x)
        
            x = self.relu(x)
            for i in range(self.reghead_layers-2):
                x = self.regHeadLayers[i](x)
                x = self.relu(x)

            x = self.endLinear(x)

        return x
