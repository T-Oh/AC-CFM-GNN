from torch_geometric.nn import  TAGConv, BatchNorm, global_mean_pool
from torch.nn import Module, LeakyReLU, Dropout, Linear, ModuleList
import torch



class TAGGraphReg(Module):
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=16, num_layers=3, dropout=.15, K = 4, reghead_size = 64, reghead_layers = 1,
                 use_batchnorm=False, use_skipcon=False):
        super(TAGGraphReg, self).__init__()
        
        self.num_layers = int(num_layers)
        self.reghead_layers = int(reghead_layers)
        self.reghead_size = int(reghead_size)
        self.dropout = Dropout(p=dropout)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.K = int(K)
        self.use_batchnorm = bool(int(use_batchnorm))
        self.use_skipcon = bool(int(use_skipcon))
        print(f'Dropout Rate {dropout}')
        print(f'K= {self.K}')
        print(f'Num Layers {self.num_layers}')
        print(f'Hidden Size {self.hidden_size}')
        print(f'Reghead Size {self.reghead_size}')
        print(f'Reghead Layers {self.reghead_layers}')


        #Convolutional Layers
        self.conv1 = TAGConv(num_node_features, self.hidden_size, bias=True,K=self.K)
        self.convLayers = ModuleList([TAGConv(self.hidden_size, self.hidden_size, bias=True, K=self.K) for i in range(self.num_layers-1)])

        #Regression Head Layers       
        self.regHead1 = Linear(self.hidden_size, self.reghead_size)
        self.regHeadLayers = ModuleList([Linear(self.reghead_size, self.reghead_size) for i in range(self.reghead_layers-2)])
        self.endLinear = Linear(self.reghead_size,num_targets,bias=True)
        self.singleLinear = Linear(self.hidden_size, num_targets)
        
        #Other Layers
        self.relu = LeakyReLU()
        self.batchnorm = BatchNorm(self.hidden_size,track_running_stats=True)
        

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index.type(torch.int64), data.edge_attr.float(), data.batch
        edge_weight = edge_weight[:,6]

        PRINT=False
        
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index)
            print(edge_weight)
            
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        if self.use_batchnorm:
            x=self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)


        #Arranging Conv Layers
        for i in range(self.num_layers -1):
            if self.use_skipcon:  
                x_ = self.convLayers[i](x, edge_index=edge_index, edge_weight=edge_weight)
                if self.use_batchnorm:
                    x_ = self.batchnorm(x_)
                x = (self.relu(x_)+x)/2
                x = self.dropout(x)
            else:
                
                x = self.convLayers[i](x, edge_index=edge_index, edge_weight=edge_weight)
                if self.use_batchnorm:
                    x = self.batchnorm(x)
                x = self.relu(x)
                x = self.dropout(x)
                
        #Pooling
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
        print(f'Output of TAGGraphReg_ {x}')
        return x
