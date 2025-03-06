
from torch.nn import Module, Linear, LeakyReLU, ModuleList, LSTM, Softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTM_LDTSF(Module):
    """
    Graph Attention Network
    """
    def __init__(self, num_features, num_targets, 
                lstm_hidden_size, num_lstm_layers,
                reghead_size, reghead_layers,
                gat_dropout, max_seq_length, task
                ):
        """
        INPUT
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

        super(LSTM_LDTSF, self).__init__()

        #Params
        self.num_lstm_layers = int(num_lstm_layers)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.reghead_layers = int(reghead_layers)
        self.reghead_size = int(reghead_size)
        self.task = task
        self.max_seq_length = max_seq_length



        #LSTM Layers
        self.lstm = LSTM(num_features, self.lstm_hidden_size, num_layers=self.num_lstm_layers, dropout=gat_dropout, batch_first=True)


        #Regression Head Layers
        if 'typeII' in self.task:
            self.regHead1 = Linear(self.lstm_hidden_size*max_seq_length, self.reghead_size)  
            self.endLinear = Linear(self.reghead_size, num_targets*max_seq_length, bias=True)
            self.singleLinear = Linear(self.lstm_hidden_size*max_seq_length, num_targets)
        else:
            self.regHead1 = Linear(self.lstm_hidden_size, self.reghead_size)
            self.endLinear = Linear(self.reghead_size, num_targets, bias=True)
            self.singleLinear = Linear(self.lstm_hidden_size, num_targets)
        self.regHeadLayers = ModuleList(Linear(self.reghead_size, self.reghead_size) for i in range(self.reghead_layers-2))

        #Additional Layers
        self.relu = LeakyReLU()
        self.softmax = Softmax(dim=0)


    def forward(self, data):
        
        
        lengths = data[2]
        x = pack_padded_sequence(data[0], lengths, batch_first=True, enforce_sorted=False).to(device='cuda')   

        PRINT=False
        if PRINT:
            print("START")
            print(x.shape)
            print(x)
            print('Graph Label shape:', data.y.shape)


        #LSTM

        x, (hn, cn) = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        #x = self.singleLinear(x[:, -1, :])
 

        if 'typeII' not in self.task:
            x = x[:, -1, :]
        else:
            print('x shape:', x.shape)
            x = x.view(x.shape[0], -1)
            print('x shape:', x.shape)
        #REGRESSION HEAD
        if self.reghead_layers == 1:
            #x = self.singleLinear(x[torch.arange(x.size(0)), lengths - 1])
            x = self.singleLinear(x)


        elif self.reghead_layers > 1:
            #x = self.regHead1(x[torch.arange(x.size(0)), lengths - 1])
            x = self.regHead1(x)
            x = self.relu(x)
            for i in range(self.reghead_layers-2):
                x = self.regHeadLayers[i](x)
                x = self.relu(x)

            x = self.endLinear(x)
        
        x = self.softmax(x)


        

        return x.to(float)
