from torch_geometric.nn import  GATv2Conv, BatchNorm
from torch.nn import Module, Dropout, Linear, LeakyReLU, ModuleList


class GAT(Module):
    """
    Graph Attention Network
    """
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1, hidden_size=1, num_layers=1, reghead_size=500, reghead_layers=1,
                 dropout=0.0, gat_dropout=0.0,  num_heads=1, use_skipcon=False, use_batchnorm=False):
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
         gat_dropout        :   float
             dropout applied within the GATv2Conv layers
         num_heads          :   int
              number of attention heads
        use_batchnorm       :   bool
             whether batchnorm should be applied
         use_skipcon        :   boo
             wether skip connections should be applied
        
        """
        
        super(GAT, self).__init__()
        
        #Params
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.reghead_size = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.num_heads = int(num_heads)
        self.use_skipcon = bool(int(use_skipcon))
        self.use_batchnorm = bool(int(use_batchnorm))

        
        
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
                x_ = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                if self.use_batchnorm:
                    x_ = self.batchnorm(x_)
                x = (self.relu(x_)+x)/2
                x = self.dropout(x)
            else:
                x = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                if self.use_batchnorm:
                    x = self.batchnorm(x)
                x = self.relu(x)
                x = self.dropout(x)
        
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
