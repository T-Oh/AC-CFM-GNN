import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class LazyTimeseriesGraphDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: Path to the directory containing sequences and timesteps.
        """
        self.root_dir = os.path.join(root_dir + 'processed/')
        self.sequence_paths = sorted(
            [entry for entry in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, entry))]
        )
        self.static_data = torch.load(os.path.join(self.root_dir, 'data_static.pt'))
        self.sequence_lengths = {}

        # Precompute sequence lengths
        for seq in self.sequence_paths:
            seq_dir = os.path.join(self.root_dir, seq)
            timesteps = sorted(os.listdir(seq_dir))  # List of timestep files
            self.sequence_lengths[seq] = len(timesteps)

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        """
        Returns all timesteps for a single sequence as a list of Data objects.
        """
        sequence_name = self.sequence_paths[idx]
        seq_dir = os.path.join(self.root_dir, sequence_name)
        timestep_files = sorted(os.listdir(seq_dir))  # Ensure timesteps are ordered

        # Load timesteps lazily
        timesteps = [self.static_data]  # Add static data as the first timestep
        for file in timestep_files:
            timestep_path = os.path.join(seq_dir, file)
            graph_data = torch.load(timestep_path)  # Load graph data for this timestep
            timesteps.append(graph_data)
        return timesteps


def collate_fn(batch):
    """
    batch: List of sequences, where each sequence is a list of timesteps (Data objects).
    """
    max_length = max(len(sequence) for sequence in batch)
    padded_sequences = []
    sequence_lengths = []

    for sequence in batch:
        # Repeat the first step (static data) to pad sequence to max_length
        padded_sequence = [sequence[0]] * (max_length - len(sequence)) + sequence
        padded_sequences.append(padded_sequence)
        sequence_lengths.append(len(sequence))

    # Convert padded sequences to batched graphs
    batched_sequences = []
    for sequence in padded_sequences:        #timestep_data = [seq[timestep] for seq in padded_sequences]
        batched_graph = Batch.from_data_list(sequence)
        batched_sequences.append(batched_graph)


    return batched_sequences, sequence_lengths


def create_lstm_dataset(root_dir):
    dataset = LazyTimeseriesGraphDataset(root_dir=root_dir)
    return dataset
def create_lstm_dataloader(dataset, batch_size, shuffle):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return loader

