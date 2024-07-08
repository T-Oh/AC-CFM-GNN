from torch_geometric.nn import  TAGConv, BatchNorm, global_mean_pool
from torch.nn import Module, LeakyReLU, Dropout, Linear, ModuleList
import torch
from torch.utils.checkpoint import checkpoint



class TAGNodeReg(Module):
    """
    Topology Adaptive Graph convolutional network
    """
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=16, num_layers=3, dropout=.15, K = 4, reghead_size = 64, reghead_layers = 1,
                 use_batchnorm=False, use_skipcon=False, task='NodeReg'):
        """
        INPUT
        num_node_features   :   int
            number of node features in data
        num_targets         :   int
            number of labels in the data
        hidden_size         :   int
            the number of hidden features to be used
        num_layers          :   int
            the number of layers to be used
        dropout             :   float
             the dropout to be applied
         K                  :   int
             number of jumps within the TAG layer
        reghead_size        :   int
            number of hidden features of the regression head
        reghead_layers      :   itn
            number of regression head layers
        use_batchnorm       :   bool
             whether batchnorm should be applied
         use_skipcon        :   boo
             wether skip connections should be applied
        
        """
        super(TAGNodeReg, self).__init__()
        
        self.num_layers = int(num_layers)
        self.reghead_layers = int(reghead_layers)
        self.reghead_size = int(reghead_size)
        self.dropout = Dropout(p=dropout)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.K = int(K)
        self.use_batchnorm = bool(int(use_batchnorm))
        self.use_skipcon = bool(int(use_skipcon))
        self.task = task



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

    def custom_activation(self, layer):
        def checkpoint_forward(*inputs):
            out = layer(inputs[0])
            return out
        return checkpoint_forward
        

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index.type(torch.int64), data.edge_attr.float(), data.batch


        PRINT=False
        
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index.shape)
            print(edge_weight.shape)
            
            
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        if self.use_batchnorm:
            x=self.batchnorm(x)
        x = checkpoint(self.custom_activation(self.relu), x)
        x = self.dropout(x)


        #Arranging Conv Layers
        for i in range(self.num_layers -1):
            if self.use_skipcon:  
                x_ = self.convLayers[i](x, edge_index=edge_index, edge_weight=edge_weight)
                if self.use_batchnorm:
                    x_ = self.batchnorm(x_)
                if i<self.num_layers-2:
                    x = (checkpoint(self.custom_activation(self.relu), x_)+x)/2
                else:
                    x = x_

                x = self.dropout(x)
            else:
                
                x = self.convLayers[i](x, edge_index=edge_index, edge_weight=edge_weight)
                if self.use_batchnorm:
                    x = self.batchnorm(x)
                x = checkpoint(self.custom_activation(self.relu), x)
                if i < self.num_layers-2:
                    x = self.dropout(x)
        
        if self.task != 'GraphReg':
        #Regression Head
            if self.reghead_layers == 1:
                    x = checkpoint(self.custom_activation(self.singleLinear), x)


            elif self.reghead_layers > 1:
                x = checkpoint(self.custom_activation(self.regHead1), x)

                x = checkpoint(self.custom_activation(self.relu), x)
                for i in range(self.reghead_layers-2):
                    x = checkpoint(self.custom_activation(self.regHeadLayers[i]), x)
                    x = checkpoint(self.custom_activation(self.relu), x)
                    #print(out)
                x = checkpoint(self.custom_activation(self.endLinear), x)
        else:
            x = global_mean_pool(x, batch)
            x = checkpoint(self.custom_activation(self.singleLinear), x)
        return x
