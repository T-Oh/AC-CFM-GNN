import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv
from torch_geometric.data import Batch

class TAG_LSTM(nn.Module):
    def __init__(self, num_node_features, conv_hidden_size, num_conv_targets, num_conv_layers, lstm_hidden_size, num_lstm_layers, reghead_size, reghead_layers,
                 dropout, K, use_skipcon, use_batchnorm, task, reghead_type='single'):
        super(TAG_LSTM, self).__init__()

        print(f'num_node_features: {num_node_features}')
        print(f'conv_hidden_size: {conv_hidden_size}')
        print(f'num_conv_targets: {num_conv_targets}')
        print(f'num_conv_layers: {num_conv_layers}')
        print(f'lstm_hidden_size: {lstm_hidden_size}')
        print(f'num_lstm_layers: {num_lstm_layers}')

        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_skipcon = use_skipcon
        self.use_batchnorm = use_batchnorm
        self.reghead_config = reghead_type

        self.num_node_features  = int(num_node_features)
        self.conv_hidden_size   = int(conv_hidden_size)
        self.num_conv_targets   = int(num_conv_targets)
        self.num_conv_layers    = int(num_conv_layers)
        self.lstm_hidden_size   = int(lstm_hidden_size)
        self.num_lstm_layers    = int(num_lstm_layers)
        self.reghead_size       = int(reghead_size)
        self.reghead_layers     = int(reghead_layers)
        self.dropout            = float(dropout)
        self.K                  = int(K)


        # TAG Layers
        self.tag_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None

        # Input layer
        if self.num_conv_layers >= 2:
            self.tag_layers.append(TAGConv(self.num_node_features, self.conv_hidden_size, K=self.K))
            if self.use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_size))

            # Hidden layers
            for _ in range(1, self.num_conv_layers - 1):
                self.tag_layers.append(TAGConv(self.conv_hidden_size, self.conv_hidden_size, K=self.K))
                if self.use_batchnorm:
                    self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_size))

            # Output layer
            self.tag_layers.append(TAGConv(self.conv_hidden_size, self.num_conv_targets, K=self.K))
            if self.use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(self.num_conv_targets))
        else:
            self.tag_layers.append(TAGConv(self.num_node_features, self.num_conv_targets, K=self.K))
            if self.use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(self.num_conv_targets))

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.num_conv_targets * 2000,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=self.dropout
        )

        # Regression Heads
        def create_reghead(input_size, output_size):
            layers = []
            for i in range(self.reghead_layers - 1):
                layers.append(nn.Linear(input_size, self.reghead_size // (i + 1)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                input_size = self.reghead_size // (i + 1)
            layers.append(nn.Linear(input_size, output_size))
            return nn.Sequential(*layers)

        if self.reghead_config == 'single':
            self.reghead = create_reghead(self.lstm_hidden_size, self.reghead_size)
            self.final_layer1 = nn.Linear(self.reghead_size, 2000)
            self.final_layer2 = nn.Linear(self.reghead_size, 2000)
            self.edge_final_layer = nn.Linear(self.reghead_size, 2 * 7064)
        elif self.reghead_config == 'node_edge':
            self.node_reghead = create_reghead(self.lstm_hidden_size, self.reghead_size)
            self.final_layer1 = nn.Linear(self.reghead_size, 2000)
            self.final_layer2 = nn.Linear(self.reghead_size, 2000)
            self.edge_reghead = create_reghead(self.lstm_hidden_size, 2 * 7064)
        elif self.reghead_config == 'node_node_edge':
            self.node_reghead1 = create_reghead(self.lstm_hidden_size, 2000)
            self.node_reghead2 = create_reghead(self.lstm_hidden_size, 2000)
            self.edge_reghead = create_reghead(self.lstm_hidden_size, 2 * 7064)

        self.relu = nn.ReLU()

    def forward(self, sequences):
        lstm_inputs = []
        batch_size = len(sequences[1])
        for sequence in sequences[0]:
            #print(sequence)
            #batched_graph = Batch.from_data_list(sequence).to(self.device)
            batched_graph = sequence.to(self.device)
            #print(batched_graph.device)
            #timestep_embeddings = []
            num_timesteps = len(sequence.x) // 2000           
            #for t in range(batch_size):
            #batch_graph = sequence[t]
            x, edge_index, edge_attr = batched_graph.x, batched_graph.edge_index, torch.norm(batched_graph.edge_attr, dim=1)

            for i, tag_layer in enumerate(self.tag_layers):
                skip_connection = x if self.use_skipcon else None
                x = tag_layer(x, edge_index, edge_weight=edge_attr)
                if self.use_batchnorm:
                    if i+1 == self.num_conv_layers: x = x.view(-1,self.num_conv_targets)
                    else:   x = x.view(-1,self.conv_hidden_size)
                    x = self.batch_norms[i](x)
                x = self.relu(x)
                if self.use_skipcon and skip_connection is not None and x.shape == skip_connection.shape:
                    x = (x + skip_connection) / 2

            x = x.view(num_timesteps, -1)
            #print('X SHAPE')
            #print(x.shape)
            lstm_inputs.append(x)
            #timestep_embeddings.append(x.reshape(-1))
            #lstm_input = torch.stack(lstm_inputs, dim=1)
            #lstm_inputs.append(lstm_input.transpose(0, 1))

        lstm_inputs = torch.stack(lstm_inputs, dim=0)
        #print('lstm_inputs shape')
        #print(lstm_inputs.shape)
        lstm_output, _ = self.lstm(lstm_inputs)
        lstm_output = self.relu(lstm_output[:, -1, :])

        if self.reghead_config == 'single':
            # Single regression head for both node and edge
            reghead_output = self.reghead(lstm_output)
            final_output1 = self.final_layer1(reghead_output)
            final_output2 = self.final_layer2(reghead_output)
            edge_output = self.edge_final_layer(reghead_output)
        elif self.reghead_config == 'node_edge':
            # Separate regression heads for node and edge
            reghead_output = self.node_reghead(lstm_output)
            final_output1 = self.final_layer1(reghead_output)
            final_output2 = self.final_layer2(reghead_output)
            edge_output = self.edge_reghead(lstm_output)
        else:
            # Separate regression heads for both node labels as well as the edge label
            final_output1 = self.node_reghead1(lstm_output)
            final_output2 = self.node_reghead2(lstm_output)
            edge_output = self.edge_reghead(lstm_output)

        final_output = torch.stack([final_output1, final_output2], dim=1)
        final_output = (final_output, edge_output.reshape(-1, 2, 7064))

        return final_output
