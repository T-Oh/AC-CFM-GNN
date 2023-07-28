from torch.nn import Module, LeakyReLU, Linear, ModuleList, Dropout, BatchNorm1d




class ridge(Module):
    """Simple ridge model to use as baseline
    DEPRECATED using run_ridge.py instead
    """

    def __init__(self, num_node_features, hidden_size=128):
        super().__init__()
        self.linear = Linear(in_features=int(num_node_features), out_features=1)


    def forward(self, data):
        x  = data.x
        print(f'Shape of data in ridge: {x.shape}')
        
        x = self.linear(x)
        #print(x)
        
        return x

class MLP(Module):
    """
    Multi layer perceptron baseline
    """
    def __init__(self, num_node_features, hidden_size, num_layers, dropout, use_batchnorm, use_skipcon):
        """
        INPUT
        num_node_features   :   int
            number of node features in data
        hidden_size         :   int
            the number of hidden features to be used
        num_layers          :   int
            the number of layers to be used
        dropout             :   float
             the dropout to be applied
        use_batchnorm       :   bool
             whether batchnorm should be applied
         use_skipcon        :   boo
             wether skip connections should be applied
        
        """
        
        super().__init__()
        #Parameters
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.use_batchnorm = bool(int(use_batchnorm))
        self.use_skipcon = bool(int(use_skipcon))
        
        #Linear Layers
        self.lin_single = Linear(int(num_node_features), 1)
        self.lin_in = Linear(in_features=int(num_node_features), out_features = int(hidden_size))
        self.layers = ModuleList([Linear(in_features=int(hidden_size), out_features = int(hidden_size)) for i in range(self.num_layers-2)])
        self.lin_end = Linear(in_features=int(hidden_size), out_features = 1)
        
        #Other Layers
        self.ReLu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        self.batchnorm = BatchNorm1d(self.hidden_size)
               
        
    def forward(self, data):
        x = data.x
        print(f'MLP Shape {x.shape}')
        
        #Single layer
        if self.num_layers == 1:
            x = self.lin_single(x)
            
        #Multiple layers
        else:           
            x = self.lin_in(x)
            if self.use_batchnorm:  
                x = self.batchnorm(x)   #Batchnormalization
            x = self.ReLu(x)        
            x = self.dropout(x)
            
            for i in range(self.num_layers-2):               
                if self.use_skipcon:   #Skip connections
                    print('Using Skipcon')
                    x_ = self.layers[i](x)
                    if self.use_batchnorm:
                        print('Using Batchnorm')
                        x_ = self.batchnorm(x_)
                    x = (self.ReLu(x_)+x)/2
                    x = self.dropout(x)
                else:                #No Skip connections
                    x = self.layers[i](x)
                    if self.use_batchnorm:
                        print('Using Batchnorm')
                        x = self.batchnorm(x)
                    x = self.ReLu(x)
                    x = self.dropout(x)
                    
            x = self.lin_end(x)
        return x
        
