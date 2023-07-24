from torch.nn import Module, LeakyReLU, Linear, ModuleList, Dropout, BatchNorm1d




class ridge_graphreg(Module):
    """Simple ridge model to use as baseline"""

    def __init__(self, num_node_features, hidden_size=128):
        super().__init__()
        self.linear = Linear(in_features=2000*int(num_node_features), out_features=1)


    def forward(self, data):
        x  = data.x
        print(f'Shape of data in ridge: {x.shape}')
        
        x = self.linear(x)
        #print(x)
        
        return x

class MLP_graphreg(Module):
    def __init__(self, num_node_features, hidden_size, num_layers, dropout, use_batchnorm, use_skipcon, batchsize):
        super().__init__()
        #Parameters
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.use_batchnorm = bool(int(use_batchnorm))
        self.use_skipcon = bool(int(use_skipcon))
        self.batchsize = int(batchsize)
        
        #Linear Layers
        self.lin_single = Linear(int(num_node_features)*batchsize*2000, 1)
        self.lin_in = Linear(in_features=int(num_node_features)*batchsize*2000, out_features = int(hidden_size))
        self.layers = ModuleList([Linear(in_features=int(hidden_size), out_features = int(hidden_size)) for i in range(self.num_layers-2)])
        self.lin_end = Linear(in_features=int(hidden_size), out_features = 1)
        
        #Other Layers
        self.ReLu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        self.batchnorm = BatchNorm1d(self.hidden_size)
               
        
    def forward(self, data):
        x = data.x
        print(f'MLP Shape {x.shape}')
        
        if self.num_layers == 1:
            x = self.lin_single(x)
            
            
        else:           
            x = self.lin_in(x)
            if self.use_batchnorm:
                x = self.batchnorm(x)
            x = self.ReLu(x)
            x = self.dropout(x)
            
            for i in range(self.num_layers-2):               
                if self.use_skipcon:   
                    print('Using Skipcon')
                    x_ = self.layers[i](x)
                    if self.use_batchnorm:
                        print('Using Batchnorm')
                        x_ = self.batchnorm(x_)
                    x = (self.ReLu(x_)+x)/2
                    x = self.dropout(x)
                else:                
                    x = self.layers[i](x)
                    if self.use_batchnorm:
                        print('Using Batchnorm')
                        x = self.batchnorm(x)
                    x = self.ReLu(x)
                    x = self.dropout(x)
                    
            x = self.lin_end(x)
        return x
        
