import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GAT_LSTM(nn.Module):
    def __init__(self, num_node_features, conv_hidden_size, num_conv_targets, num_conv_layers, lstm_hidden_size, num_lstm_layers, reghead_size, reghead_layers,
                 dropout, gat_dropout, num_heads, use_skipcon, use_batchnorm, max_seq_length, task):
        super(GAT_LSTM, self).__init__()

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

        self.num_node_features  = int(num_node_features)
        self.conv_hidden_size   = int(conv_hidden_size)
        self.num_conv_targets   = int(num_conv_targets)
        self.num_conv_layers    = int(num_conv_layers)
        self.lstm_hidden_size   = int(lstm_hidden_size)
        self.num_lstm_layers    = int(num_lstm_layers)
        self.reghead_size       = int(reghead_size)
        self.reghead_layers     = int(reghead_layers)
        self.dropout            = float(dropout)
        self.gat_dropout        = float(gat_dropout)
        self.num_heads          = int(num_heads)
        self.max_seq_length     = int(max_seq_length)

        # GAT Layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None

        # Input layer
        self.gat_layers.append(GATv2Conv(self.num_node_features, self.conv_hidden_size, heads=self.num_heads, edge_dim=2, dropout=self.gat_dropout))
        if self.use_batchnorm:
            self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_size * self.num_heads))

        # Hidden layers
        for _ in range(1, self.num_conv_layers - 1):
            self.gat_layers.append(GATv2Conv(self.conv_hidden_size * self.num_heads, self.conv_hidden_size, heads=self.num_heads, edge_dim=2, dropout=self.gat_dropout))
            if self.use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(self.conv_hidden_size * self.num_heads))

        # Output layer
        self.gat_layers.append(GATv2Conv(self.conv_hidden_size * self.num_heads, self.num_conv_targets, heads=self.num_heads, edge_dim=2, dropout=self.gat_dropout))
        if self.use_batchnorm:
            self.batch_norms.append(nn.BatchNorm1d(self.num_conv_targets * self.num_heads))

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.num_conv_targets * self.num_heads * 2000,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=self.dropout
        )

        # Regression Head
        self.reghead_layers_list = []
        input_size = self.lstm_hidden_size * self.max_seq_length if self.task == 'pred_typeII' else self.lstm_hidden_size
        for i in range(self.reghead_layers - 1):
            self.reghead_layers_list.append(nn.Linear(input_size, self.reghead_size // (i + 1)))
            self.reghead_layers_list.append(nn.ReLU())
            self.reghead_layers_list.append(nn.Dropout(self.dropout))
            input_size = self.reghead_size // (i + 1)
        if self.task == 'NodeReg':
            self.final_layer1 = nn.Linear(input_size, 2000)  # Final output layer 
            self.final_layer2 = nn.Linear(input_size, 2000)  # Final output layer    
        else:
            self.reghead_layers_list.append(nn.Linear(input_size, 1))  # Final output layer
        self.fc = nn.Sequential(*self.reghead_layers_list)

        self.relu = nn.ReLU()

    def forward(self, sequences):
        """
        sequences: List of batched graph sequences. Each element in the list corresponds
                   to a sequence (batched graphs at each timestep).
        """
        lstm_inputs = []  # To store the GAT embeddings for each timestep

        for sequence in sequences[0]:
            timestep_embeddings = []
            batch_size = len(sequence.x) // 2000

            for t in range(batch_size):
                batch_graph = sequence[t]
                x, edge_index, edge_attr = batch_graph.x.to(self.device), batch_graph.edge_index.to(self.device), batch_graph.edge_attr.to(self.device)


                # GAT forward pass through all GAT layers
                for i, gat_layer in enumerate(self.gat_layers):
                    skip_connection = x if self.use_skipcon else None
                    x = gat_layer(x, edge_index, edge_attr=edge_attr)
                    if self.use_batchnorm:
                        x = self.batch_norms[i](x)
                    x = self.relu(x)

                    # Add skip connection
                    if self.use_skipcon and skip_connection is not None and x.shape == skip_connection.shape:
                        print(f'i: {i}')
                        x = (x + skip_connection)/2

                timestep_embeddings.append(x.reshape(-1))
                del x, edge_index, edge_attr, batch_graph

            # Stack timestep embeddings into a sequence tensor
            lstm_input = torch.stack(timestep_embeddings, dim=1)
            lstm_inputs.append(lstm_input.transpose(0, 1))
        del sequences, timestep_embeddings, lstm_input

        # Concatenate all sequence tensors into a batch for the LSTM
        lstm_inputs = torch.stack(lstm_inputs, dim=0)

        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_inputs)
        lstm_output = self.relu(lstm_output)

        # Pass the LSTM output through the fully connected layers
        final_output = self.fc(lstm_output[:, -1, :])
        if self.task == 'NodeReg':
            final_output1 = self.final_layer1(final_output)
            final_output2 = self.final_layer2(final_output)
            final_output = torch.stack((final_output1, final_output2), dim=-1)
        return final_output
