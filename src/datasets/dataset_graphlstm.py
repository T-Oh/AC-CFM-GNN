import os
import torch

from torch.utils.data import  DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Batch

from utils.utils import create_train_test_split_GTSF

class dataset_graphlstm(Dataset):
    def __init__(self, root, sequence_indices=None, max_seq_len=100, autoregressive = True, normalized=False):
        """
        root_dir: Path to the directory containing sequences and timesteps.
        sequence_indices: Indices of sequences to include in this dataset.
        """
        self.AUTOREGRESSIVE = autoregressive
        self.WINDOWSIZE = max_seq_len
        self.root = root
        self.normalized = normalized
        print(self.normalized)

        if not self.AUTOREGRESSIVE:
            self.sequence_paths = sorted(
                [entry for entry in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, entry))]
            )
        else:
            print('Autoregressive data loading')
            self.sequence_paths = sorted(
                [entry for entry in os.listdir(self.processed_dir)[:self.WINDOWSIZE] if os.path.isdir(os.path.join(self.processed_dir, entry))]
            )
        if self.normalized: self.static_data = torch.load(os.path.join(self.root, 'normalized/data_static.pt'))
        else:   self.static_data = torch.load(os.path.join(self.processed_dir, 'data_static.pt'))
        self.max_seq_len = max_seq_len

        # Filter sequences by indices
        if sequence_indices is not None:
            self.sequence_paths = [self.sequence_paths[i] for i in sequence_indices]

    @property
    def processed_dir(self):

        if self.normalized:
            processed_dir = os.path.join(self.root, 'normalized/')
            print('Entred normalized dir')
            print(processed_dir)
        else:
            processed_dir = os.path.join(self.root, 'processed/')
        return processed_dir

    @property
    def raw_file_names(self):
        return os.listdir(self.root + "/raw")

    @property
    def processed_file_names(self):
        files = []
        for root, _, filenames in os.walk(self.root + "/processed"):
            for filename in filenames:
                if filename.startswith("data"):
                    files.append(os.path.relpath(os.path.join(root, filename), self.root + "/processed"))
        return files
      


    def __len__(self):
        return len(self.sequence_paths)
    
    
    def __getitem__(self, idx):
        """
        Returns all timesteps for a single sequence as a list of Data objects.
        """
        #print('THIS')
        print(self.processed_dir)
        sequence_name = self.sequence_paths[idx]
        seq_dir = os.path.join(self.processed_dir, sequence_name)
        timestep_files = sorted(os.listdir(seq_dir), key=lambda x: int(x.split('_')[2].split('.')[0]))
        seq_len = len(timestep_files)
        #print('seq_len:', seq_len)
        #print('max_seq_len:', self.max_seq_len)
        #If the sequence is short than max_seq_len, pad with one of the static data. The rest of the padding (if necessary) will be done in collate_fn
        if seq_len < self.max_seq_len:
            timesteps = [self.static_data.clone()]  # Add static data as the first timestep
            timesteps[0].x = timesteps[0].x[:, :4]  # Keep only the first 4 node features
        else:
            timesteps = []

        if not self.AUTOREGRESSIVE:
            for i in range(min(seq_len, self.max_seq_len)):
                if seq_len >= self.max_seq_len:
                    timestep_path = os.path.join(seq_dir, timestep_files[-self.max_seq_len+i])
                else:
                    timestep_path = os.path.join(seq_dir, timestep_files[i])
                graph_data = torch.load(timestep_path)  # Load graph data for this timestep
                graph_data.x = graph_data.x[:, :4]  # Slice node features to keep only the first 4
                timesteps.append(graph_data)
        else:
            for i in range(min(seq_len, self.max_seq_len)):
                timestep_path = os.path.join(seq_dir, timestep_files[i])
                graph_data = torch.load(timestep_path)  # Load graph data for this timestep
                graph_data.x = graph_data.x[:, :4]  # Slice node features to keep only the first 4
                timesteps.append(graph_data)
        #print(timesteps)
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



def create_lstm_datasets(cfg, normalized=False):
    dataset = dataset_graphlstm(root=cfg["dataset::path"], max_seq_len=cfg["max_seq_length"], autoregressive=cfg["autoregressive"], normalized=normalized)
    train_indices, test_indices = create_train_test_split_GTSF(dataset, cfg["train_size"], cfg["manual_seed"], cfg["stormsplit"])
    trainset = dataset_graphlstm(root=cfg['dataset::path'], sequence_indices=train_indices, max_seq_len=cfg["max_seq_length"], autoregressive=cfg["autoregressive"], normalized=normalized)
    testset = dataset_graphlstm(root=cfg['dataset::path'], sequence_indices=test_indices, max_seq_len=cfg["max_seq_length"], autoregressive=cfg["autoregressive"], normalized=normalized)

    return trainset, testset

def create_lstm_dataloader(dataset, batch_size, shuffle, pin_memory, num_workers):   #indices,
    """
    Creates a DataLoader for the given dataset and indices.
    """
    #subset = dataset_graphlstm(dataset.root_dir, sequence_indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=False)
    return loader
