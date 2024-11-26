import h5py
import numpy as np

def load_bus_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # Access the 'bus' dataset
        bus_data_ref = f['clusterresult_']['bus']
        print(bus_data_ref.shape)
        
        # Loop through and dereference each object reference in the 'bus' dataset
        bus_data = []

        ref = bus_data_ref[0,0]  # Get the object reference
        dereferenced_data = f[ref]  # Dereference it
        bus_data.append(dereferenced_data[()])  # Append the actual data
        
        return bus_data

# Example usage
file_path = '/home/tohlinger/HUI/Documents/hi-accf-ml/raw/clusterresults_1204.mat'
bus_data = load_bus_data(file_path)
print(np.array(bus_data).shape)