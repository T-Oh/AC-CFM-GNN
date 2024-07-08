from torch_geometric.nn import BatchNorm, GINEConv, global_mean_pool
from torch.nn import Module, Dropout, Linear, BatchNorm1d, LeakyReLU
import torch.nn as nn
from torch.utils.checkpoint import checkpoint




class GINE(Module):
    """
    Graph Isomorphism Network with Edge Features
    """
    
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reghead_size=500, reghead_layers =2,
                 dropout=0.0, use_skipcon=False, use_batchnorm=True, task='NodeReg'):       
        """
        INPUT
        num_node_features   :   int
            number of node features in data
        num_edge_features   :   int
            number of edge features in the data
        num_targets         :   int
            number of labels in the data
        hidden_size         :   int
            the number of hidden features to be used
        num_layers          :   int
            the number of layers to be used
        reghead_size        :   int
            number of hidden features of the regression head
        reghead_layers      :   itn
            number of regression head layers
        dropout             :   float
             the dropout to be applied
         num_heads          :   int
              number of attention heads
        use_batchnorm       :   bool
             whether batchnorm should be applied - not implemented for GINE
         use_skipcon        :   boo
             wether skip connections should be applied
        
        """
        
        super(GINE, self).__init__()
        #use_batchnorm does not have an effect on GINE since the batchnorm is automatically used inside the layers

        #Params


        self.num_layers=int(num_layers)
        self.use_skipcon = bool(int(use_skipcon))
        self.reghead_layers = int(reghead_layers)
        self.task = task
        hidden_size = int(hidden_size)
        reghead_size = int(reghead_size)
        
        print('Using Gine with ', num_node_features, ' Node features')
        print('and', num_edge_features, 'Edge Features\n')
        
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
        self.singleLinear = Linear(hidden_size, num_targets, bias=True)
        self.regHeadLayers = nn.ModuleList(Linear(reghead_size, reghead_size) for i in range(self.reghead_layers-2))
        self.endLinear = Linear(reghead_size,num_targets, bias=True)

        
        #Additional Layers
        self.relu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        print(f'Dropoutrate {dropout}')
        self.batchnorm = BatchNorm(hidden_size,track_running_stats=True)

    #Custom Forward Functions for Checkpointing
    def run_func_factory_conv(self, layer):
        def checkpoint_forward(*inputs):
            out = layer(inputs[0],edge_index=inputs[1], edge_attr=inputs[2])
            return out
        return checkpoint_forward 
    
    def custom_activation(self, layer):
        def checkpoint_forward(*inputs):
            out = layer(inputs[0])
            return out
        return checkpoint_forward

    def forward(self, data):
        
        
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index, data.edge_attr.float()
        if edge_weight.dim() == 1:  #If onyl 1 edge attr an empty 2nd dimension needs to be added
            edge_weight = edge_weight.unsqueeze(1)
        PRINT=False
        if PRINT:
            print('\n\nGINE FORWARD OUTPUT\n ')
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)


        x = self.convLayer1(x, edge_index, edge_weight)
        x = checkpoint(self.custom_activation(self.relu), x)
        x = self.dropout(x)


        #Arranging Conv Layers
        if self.use_skipcon:   
            if PRINT: print('Using Skipcon')
            for i in range(self.num_layers -1):
                x_ = self.gines[i](x, edge_index, edge_weight)
                if i < self.num_layers-2:   x = (checkpoint(self.custom_activation(self.relu), x_)+x)/2
                else:                       x = x_
                x = self.dropout(x)

        else:
            for i in range(self.num_layers-1):
                x = self.gines[i](x, edge_index, edge_weight)
                if i < self.num_layers-2:
                    x = checkpoint(self.custom_activation(self.relu), x)
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

        
        #print(x)
        #print("END")
        return x
