import torch
import torch.nn as nn

class LSTM_Baseline(nn.Module):
    def __init__(self, num_node_features, lstm_hidden_size, num_lstm_layers, reghead_size, reghead_layers,
                 dropout, task, reghead_type='single'):
        super(LSTM_Baseline, self).__init__()

        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reghead_config = reghead_type

        self.num_node_features = int(num_node_features)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.num_lstm_layers = int(num_lstm_layers)
        self.reghead_size = int(reghead_size)
        self.reghead_layers = int(reghead_layers)
        self.dropout = float(dropout)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.num_node_features * 2000,  # flattened node features
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
            sequence = sequence.to(self.device)
            x = sequence.x  # shape: [num_timesteps * 2000, num_node_features]
            num_timesteps = x.shape[0] // 2000

            x = x.view(num_timesteps, 2000, self.num_node_features)
            x = x.reshape(num_timesteps, -1)  # [timesteps, 2000 * num_node_features]

            lstm_inputs.append(x)

        lstm_inputs = torch.stack(lstm_inputs, dim=0)  # [batch, timesteps, features]
        lstm_output, _ = self.lstm(lstm_inputs)
        lstm_output = self.relu(lstm_output[:, -1, :])  # last timestep

        if self.reghead_config == 'single':
            reghead_output = self.reghead(lstm_output)
            final_output1 = self.final_layer1(reghead_output)
            final_output2 = self.final_layer2(reghead_output)
            edge_output = self.edge_final_layer(reghead_output)
        elif self.reghead_config == 'node_edge':
            reghead_output = self.node_reghead(lstm_output)
            final_output1 = self.final_layer1(reghead_output)
            final_output2 = self.final_layer2(reghead_output)
            edge_output = self.edge_reghead(lstm_output)
        else:
            final_output1 = self.node_reghead1(lstm_output)
            final_output2 = self.node_reghead2(lstm_output)
            edge_output = self.edge_reghead(lstm_output)

        final_output = torch.stack([final_output1, final_output2], dim=1)
        final_output = (final_output, edge_output.reshape(-1, 2, 7064))

        return final_output
