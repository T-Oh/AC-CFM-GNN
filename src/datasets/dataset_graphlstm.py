import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch

class dataset_graphlstm(Dataset):
    def __init__(self, root_dir, sequence_indices=None, max_seq_len=100):
        """
        root_dir: Path to the directory containing sequences and timesteps.
        sequence_indices: Indices of sequences to include in this dataset.
        """
        # Ensure the root directory ends with 'processed/' without duplication
        if not root_dir.endswith('processed/'):
            self.root_dir = os.path.join(root_dir, 'processed/')
        else:
            self.root_dir = root_dir

        self.sequence_paths = sorted(
            [entry for entry in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, entry))]
        )
        self.static_data = torch.load(os.path.join(self.root_dir, 'data_static.pt'))
        self.max_seq_len = max_seq_len

        # Filter sequences by indices
        if sequence_indices is not None:
            self.sequence_paths = [self.sequence_paths[i] for i in sequence_indices]


    def __len__(self):
        return len(self.sequence_paths)
    
    
    def __getitem__(self, idx):
        """
        Returns all timesteps for a single sequence as a list of Data objects.
        """
        sequence_name = self.sequence_paths[idx]
        seq_dir = os.path.join(self.root_dir, sequence_name)
        timestep_files = sorted(os.listdir(seq_dir), key=lambda x: int(x.split('_')[2].split('.')[0]))
        seq_len = len(timestep_files)
        #If the sequence is short than max_seq_len, pad with one of the static data. The rest of the padding (if necessary) will be done in collate_fn
        if seq_len < self.max_seq_len:
            timesteps = [self.static_data.clone()]  # Add static data as the first timestep
            timesteps[0].x = timesteps[0].x[:, :4]  # Keep only the first 4 node features
        else:
            timesteps = []

        for i in range(min(seq_len, self.max_seq_len)):
            if seq_len >= self.max_seq_len:
                timestep_path = os.path.join(seq_dir, timestep_files[-self.max_seq_len+i])
            else:
                timestep_path = os.path.join(seq_dir, timestep_files[i])

            graph_data = torch.load(timestep_path)  # Load graph data for this timestep
            graph_data.x = graph_data.x[:, :4]  # Slice node features to keep only the first 4
            timesteps.append(graph_data)
        """for file in timestep_files:
            timestep_path = os.path.join(seq_dir, file)
            print(timestep_path)
            graph_data = torch.load(timestep_path)  # Load graph data for this timestep
            graph_data.x = graph_data.x[:, :4]  # Slice node features to keep only the first 4
            timesteps.append(graph_data)"""
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

def create_train_test_split(dataset, train_ratio=0.8, random_seed=42, stormsplit=0):
    """
    Splits the dataset into train and test sets at the sequence level,
    ensuring sequences where the first digit of the first integer in the folder name 
    matches stormsplit go to the test set.
    """
    #Random train test split according to train ratio if stormsplit==0
    if stormsplit == 0:
        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size
        torch.manual_seed(random_seed)
        train_indices, test_indices = random_split(range(dataset_size), [train_size, test_size])
    else:
        # Extract folder names from dataset paths
        folder_names = [os.path.basename(path) for path in dataset.sequence_paths]

        # Separate train and test indices based on stormsplit condition
        train_indices, test_indices = [], []

        for i, folder in enumerate(folder_names):
            # Extract first integer from folder name (assuming 'scenario_{int}_{int}')
            first_number = folder.split('_')[1]  # Extract first integer part
            first_digit = int(first_number[0])   # Extract first digit
            if first_digit == stormsplit:
                test_indices.append(i)  # Assign to test set
            else:
                train_indices.append(i)  # Assign to train set


    return train_indices, test_indices

def create_lstm_datasets(root_dir, train_ratio, random_seed, stormsplit, max_seq_len):
    dataset = dataset_graphlstm(root_dir=root_dir, max_seq_len=max_seq_len)
    train_indices, test_indices =create_train_test_split(dataset, train_ratio, random_seed, stormsplit)
    trainset = dataset_graphlstm(dataset.root_dir, sequence_indices=train_indices, max_seq_len=max_seq_len)
    testset = dataset_graphlstm(dataset.root_dir, sequence_indices=test_indices, max_seq_len=max_seq_len)

    return trainset, testset

def create_lstm_dataloader(dataset, batch_size, shuffle, pin_memory, num_workers):   #indices,
    """
    Creates a DataLoader for the given dataset and indices.
    """
    #subset = dataset_graphlstm(dataset.root_dir, sequence_indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=True)
    return loader
