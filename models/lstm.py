from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Module, Dropout, Linear, LeakyReLU, ModuleList, LSTM
from torch.nn.utils.rnn import pack_padded_sequence


class GCNLSTM(Module):
    """
    Graph Attention Network
    """
    def __init__(self, num_node_features, conv_hidden_size, num_conv_targets, num_conv_layers,
                lstm_hidden_size, num_lstm_layers,
                reghead_size, reghead_layers, num_targets,
                dropout, gat_dropout, use_skipcon, use_batchnorm, 
                len_sequence, num_nodes=2000):
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
        use_batchnorm       :   bool
             whether batchnorm should be applied
         use_skipcon        :   boo
             wether skip connections should be applied

        """

        super(GCNLSTM, self).__init__()

        #Params
        self.num_conv_layers = int(num_conv_layers)
        self.conv_hidden_size = int(conv_hidden_size)
        self.num_conv_targets = int(num_conv_targets)
        self.num_lstm_layers = int(num_lstm_layers)
        self.reghead_layers = int(reghead_layers)
        self.use_skipcon = bool(int(use_skipcon))
        self.use_batchnorm = bool(int(use_batchnorm))
        self.num_nodes = int(num_nodes)



        #Conv Layers
        if self.num_conv_layers == 1:
            self.conv1 = GCNConv(num_node_features, num_conv_targets)
        else:
            self.conv1=GCNConv(num_node_features,conv_hidden_size)
            self.convLayers = ModuleList([GCNConv(conv_hidden_size, conv_hidden_size) for i in range(num_conv_layers-2)])
            self.convFinal = GCNConv(conv_hidden_size, num_conv_targets)

        #LSTM Layers
        self.lstm = LSTM(num_conv_targets*num_nodes, lstm_hidden_size, num_layers=num_lstm_layers, dropout=gat_dropout, batch_first=True)


        #Regression Head Layers
        self.regHead1 = Linear(lstm_hidden_size, reghead_size)
        self.singleLinear = Linear(lstm_hidden_size, num_targets)
        self.regHeadLayers = ModuleList(Linear(reghead_size, reghead_size) for i in range(self.reghead_layers-2))
        self.endLinear = Linear(reghead_size,num_targets,bias=True)

        #Additional Layers
        self.relu = LeakyReLU()
        self.dropout = Dropout(p=dropout)
        self.batchnorm = BatchNorm(conv_hidden_size, track_running_stats=True)


    def forward(self, data):

        x, edge_index= data.x, data.edge_index

        PRINT=False
        if PRINT:
            print("START")
            print(x.shape)
            print(edge_index.shape)
            print(x)
            print(edge_index)
            print('Graph Label shape:', data.y.shape)
            #print(edge_weight)


        #GNN
        #Initial Conv Layer
        x = self.conv1(x, edge_index=edge_index)
        if self.use_batchnorm:
            x=self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)

        for i in range(self.num_conv_layers-1):
            #Pass throught layer
            if i == self.num_conv_layers-2: x_ = self.convFinal(x, edge_index=edge_index)                       #Final Conv Layer
            else:                           x_ = self.convLayers[i](x, edge_index=edge_index)                   #Hidden Conv Layers
            #Batchnorm
            if self.use_batchnorm and i<self.num_conv_layers-2:  x_ = self.batchnorm(x_)
            #SkipCon
            if self.use_skipcon and i<self.num_conv_layers-2:   x = (self.relu(x_)+x)/2
            else:                                               x = self.relu(x_)
            #Dropout
            x = self.dropout(x)


        #LSTM
        x = x.reshape(int(x.shape[0]/self.num_nodes),self.num_nodes*x.shape[1]).unsqueeze(0)    #Reshaping for LSTM
        _, (h_n, _) = self.lstm(x)  # Apply LSTM
        h_n = h_n[-1]  # Consider only the last layer of the LSTM  

        x = self.relu(h_n.reshape(-1))
        #REGRESSION HEAD
        if self.reghead_layers == 1:
                x = self.singleLinear(x)


        elif self.reghead_layers > 1:
            x = self.regHead1(x)

            x = self.relu(x)
            for i in range(self.reghead_layers-2):
                x = self.regHeadLayers[i](x)
                x = self.relu(x)

            x = self.endLinear(x)
        print('Final Output Shape:', x.shape)

        return x
