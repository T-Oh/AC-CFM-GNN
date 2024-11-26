import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GAT_LSTM(nn.Module):
    def __init__(self, num_node_features, conv_hidden_size, num_conv_targets, num_conv_layers, lstm_hidden_size, num_lstm_layers):
        super(GAT_LSTM, self).__init__()

        print(f'num_node_features: {num_node_features}')
        print(f'conv_hidden_size: {conv_hidden_size}')
        print(f'num_conv_targets: {num_conv_targets}')
        print(f'num_conv_layers: {num_conv_layers}')
        print(f'lstm_hidden_size: {lstm_hidden_size}')
        print(f'num_lstm_layers: {num_lstm_layers}')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # GAT Layer(s)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(num_node_features, conv_hidden_size, heads=1, edge_dim=2))
        
        for _ in range(1, num_conv_layers - 1):
            self.gat_layers.append(GATv2Conv(conv_hidden_size, conv_hidden_size, heads=1, edge_dim=2))
        
        self.gat_layers.append(GATv2Conv(conv_hidden_size, num_conv_targets, heads=1, edge_dim=2))
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=num_conv_targets*2000,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden_size, 1)  # Example output layer
    
    def forward(self, sequences):
        """
        sequences: List of batched graph sequences. Each element in the list corresponds
                   to a sequence (batched graphs at each timestep).
        """
        lstm_inputs = []  # To store the GAT embeddings for each timestep

        #print(f'sequences shape: {sequences}')
        for sequence in sequences[0]:
            # `sequence` is a batched graph for one sequence
            timestep_embeddings = []
            batch_size = len(sequence.x) // 2000
            for t in range(batch_size):
                # Extract graph at timestep `t` (indexed in the Batch object)
                batch_graph = sequence[t]
                x, edge_index, edge_attr = batch_graph.x.to(self.device), batch_graph.edge_index.to(self.device), batch_graph.edge_attr.to(self.device)
                # GAT forward pass through all GAT layers
                for gat_layer in self.gat_layers:
                    x = gat_layer(x, edge_index, edge_attr=edge_attr)
                timestep_embeddings.append(x.reshape(-1))
                del x, edge_index, edge_attr, batch_graph
            
            # Stack timestep embeddings into a sequence tensor
            lstm_input = torch.stack(timestep_embeddings, dim=1)  # Shape: [batch_size, sequence_length, gat_hidden_dim]
            #print(f'lstm_input shape: {lstm_input.shape}')

            lstm_inputs.append(lstm_input.transpose(0,1))
        del sequences, timestep_embeddings, lstm_input
            
        
        # Concatenate all sequence tensors into a batch for the LSTM
        #lstm_inputs = torch.cat(lstm_inputs, dim=2)  # Shape: [num_sequences, sequence_length, gat_hidden_dim]
        lstm_inputs = torch.stack(lstm_inputs, dim=0)  # Shape: [1, num_sequences * sequence_length, gat_hidden_dim]
        

        #print(f'lstm_inputs shape: {lstm_inputs.shape}')
        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_inputs)  # Shape: [num_sequences, sequence_length, lstm_hidden_dim]
        #print(f'lstm_output shape: {lstm_output.shape}')
        # Take the last output of the LSTM (or apply additional layers if needed)
        final_output = self.fc(lstm_output[:, -1, :])  # Shape: [num_sequences, output_dim]
        #print(f'final_output shape: {final_output.shape}')
        return final_output
