from torch_geometric.nn import Sequential, TAGConv, global_add_pool, global_mean_pool
from torch.nn import Module, ReLU, Dropout, Sigmoid, Linear, Tanh
import torch


class TAG(Module):
    """Topology adaptive graph convolutional model with pooling layer for graph classification"""

    def __init__(self, num_node_features=2, num_targets = 1, hidden_size=64, num_layers=3, dropout=.5):
        super().__init__()
        layers = []

        for i in range(num_layers - 1):
            if not layers:
                layers.append((TAGConv(num_node_features, hidden_size), 'x, edge_index, edge_weight -> x1'))
            else:
                layers.append((ReLU(), f'x{i} -> x{i}a'))
                layers.append((Dropout(p=dropout), f'x{i}a -> x{i}d'))
                layers.append((TAGConv(hidden_size, hidden_size), f'x{i}d, edge_index, edge_weight -> x{i+1}'))

        layers.append((ReLU(), f'x{num_layers - 1} -> x{num_layers - 1}a'))
        layers.append((Dropout(p=dropout), f'x{num_layers - 1}a -> x{num_layers - 1}d'))
        layers.append((TAGConv(hidden_size, num_targets), f'x{num_layers - 1}d, edge_index, edge_weight -> x{num_layers}'))

        layers.append((global_add_pool, f'x{num_layers}, batch -> y'))
        layers.append((Sigmoid(), f'y -> y_out'))

        self.model = Sequential('x, batch, edge_index, edge_weight', layers)
        self.model.to(float)

    def forward(self, data):
        x, batch, edge_index, edge_weight  = data.x, data.batch, data.edge_index, data.edge_attr
        return self.model(x, batch, edge_index, edge_weight)


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


class TAGNodeReg(Module):
    def __init__(self, num_node_features=2, num_targets=1, hidden_size=16, num_layers=3, dropout=.15):
        super(TAGNodeReg, self).__init__()
        self.convsingle = TAGConv(num_node_features, 1,bias=True,K=4).to(float)
        self.conv1 = TAGConv(num_node_features,hidden_size,bias=True,K=4)
        
        self.conv2 = TAGConv(hidden_size, hidden_size,bias=True,K=4).to(float)
        self.endLinear = Linear(hidden_size,num_targets,bias=False).to(float)
        self.endSigmoid = Sigmoid()
        self.endTanh=Tanh()
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.num_layers = num_layers

    def forward(self, data):
        x, batch, edge_index, edge_weight = data.x, data.batch, data.edge_index.type(torch.int64), data.edge_attr.float()

        PRINT=False
        
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index)
            print(edge_weight)
            
        #x = self.convsingle(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x=self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        print('XSHAPE AFTER CONV1')
        print(x.shape)

        for _ in range(self.num_layers - 2):
            x = self.relu(x)
            #print(x)
            #x = self.dropout(x)
            #print(x)
            x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
            print(x)

        x = self.relu(x)
        #print(x)
        #x = self.dropout(x)
        #print(x)
        x=self.endLinear(x)
        print(f'SHAPE AFTER ENDLINEAR {x.shape}')
        #print("Pool")
        #print(x)
        x = self.endSigmoid(x)
        x.type(torch.FloatTensor)
        #print(x)
        print("END")
        return x


class TAGTest(Module):
    def __init__(self, batch=True, num_node_features=2, num_targets=1, hidden_size=64, num_layers=3, dropout=.0):
        print(num_layers)
        super(TAGTest, self).__init__()
        self.conv1 = TAGConv(num_node_features, hidden_size).to(float)
        self.conv2 = TAGConv(hidden_size, hidden_size).to(float)
        self.endconv = TAGConv(hidden_size, num_targets).to(float)
        self.endSigmoid = Sigmoid()
        self.pool = global_mean_pool
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.num_layers = num_layers
        self.batch_training = batch
        self.sigmoid = Sigmoid()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.float()

        if self.batch_training:
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.int64)
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.endconv(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.pool(x, batch)
        #x = self.sigmoid(x)
        return x