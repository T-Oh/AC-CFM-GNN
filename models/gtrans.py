from torch_geometric.nn import  TransformerConv, BatchNorm, global_mean_pool
from torch.nn import Module, Dropout, Linear, LeakyReLU, ModuleList
from torch.utils.checkpoint import checkpoint


class GraphTransformer(Module):
    """
    Graph Attention Network
    """
    def __init__(self, num_node_features=2, num_edge_features=7, num_targets=1,
                hidden_size=1, num_layers=1, reghead_size=500, reghead_layers=1,
                dropout=0.0, gat_dropout=0.0,  num_heads=1, use_skipcon=False,
                use_batchnorm=False, checkpoint=True, task='NodeReg'):
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

        super(GraphTransformer, self).__init__()

        #Params
        self.num_layers     = int(num_layers)
        self.hidden_size    = int(hidden_size)
        self.reghead_size   = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.num_heads      = int(num_heads)
        self.use_skipcon    = bool(int(use_skipcon))
        self.use_batchnorm  = bool(int(use_batchnorm))
        self.checkpoint     = checkpoint
        self.task           = task



        #Conv Layers
        self.conv1=TransformerConv(num_node_features,self.hidden_size, edge_dim = num_edge_features, dropout = gat_dropout, heads=self.num_heads)
        self.convLayers = ModuleList([TransformerConv(self.hidden_size*self.num_heads,self.hidden_size, edge_dim = num_edge_features, dropout = gat_dropout, heads=self.num_heads) for i in range(self.num_layers-1)])

        #Regression Head Layers
        self.regHead1 = Linear(self.hidden_size*self.num_heads, self.reghead_size)
        self.singleLinear = Linear(self.hidden_size*self.num_heads, num_targets)
        self.regHeadLayers = ModuleList(Linear(self.reghead_size, self.reghead_size) for i in range(self.reghead_layers-2))
        self.endLinear = Linear(self.reghead_size,num_targets,bias=True)

        #Additional Layers
        self.relu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        self.batchnorm = BatchNorm(self.hidden_size*self.num_heads,track_running_stats=True)

    #Checkpointing functions:
    #Function used to create run functions necessary for checkpoints
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

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr.float(), data.batch
        x.requires_grad=True
        print('x requires_grad: ', x.requires_grad)
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(1)

        PRINT=False
        if PRINT:
            print("START")
            print(x)
            print(edge_index)
            print(edge_weight)


        x = checkpoint(self.run_func_factory_conv(self.conv1), x, edge_index, edge_weight)
        if self.use_batchnorm:
            x=self.batchnorm(x)
        x = checkpoint(self.custom_activation(self.relu), x)
        x = self.dropout(x)


        #Arranging Conv Layers
        for i in range(self.num_layers -1):
            if self.use_skipcon:
                if self.checkpoint:
                    print('CHECKPOINTING')
                    x_ = checkpoint(self.run_func_factory_conv(self.convLayers[i]), x, edge_index, edge_weight)
                else:
                    x_ = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                
                if self.use_batchnorm:
                    x_ = self.batchnorm(x_)
                x = (checkpoint(self.custom_activation(self.relu),x_)+x)/2
                x = self.dropout(x)
            else:
                if self.checkpoint:                    
                    x = checkpoint(self.run_func_factory_conv(self.convLayers[i]), x, edge_index, edge_weight)
                else:
                    x = self.convLayers[i](x, edge_index=edge_index, edge_attr=edge_weight)
                if self.use_batchnorm:
                    x = self.batchnorm(x)
                x = checkpoint(self.custom_activation(self.relu),x)
                x = self.dropout(x)


        if self.task != 'GraphReg':

            if self.reghead_layers == 1:
                if self.checkpoint:
                    x = checkpoint(self.custom_activation(self.sinlgeLinear))
                else:
                    x = self.singleLinear(x)


            elif self.reghead_layers > 1:
                if self.checkpoint:
                    x = checkpoint(self.custom_activation(self.regHead1), x)
                else:
                    x = self.regHead1(x)

                x = self.relu(x)
                for i in range(self.reghead_layers-2):
                    if self.checkpoint:
                        x = checkpoint(self.custom_activation(self.regHeadLayers[i][i]), x)
                    else:
                        x = self.regHeadLayers[i](x)
                    x = self.relu(x)

                x = checkpoint(self.custom_activation(self.endLinear), x)
        else:
            x = global_mean_pool(x,batch)
            x = checkpoint(self.custom_activation(self.singleLinear), x)

        return x
