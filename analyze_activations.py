import h5py
import numpy as np

with h5py.File('llama_activations.h5', 'r') as f:
    # Print the structure of the file
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    f.visititems(print_structure)

    # Load and print data from a specific dataset
    # Replace 'layer_0/batch_0' with the path to the dataset you want to examine
    dataset_path = 'layer_15/batch_0'
    if dataset_path in f:
        data = f[dataset_path][:]
        print(f"\nData from {dataset_path}:")
        print(f"Shape: {data.shape}")
        print("First few elements:")
        print(data.flatten()[:10])  # Print first 10 elements
        print(f"Mean: {np.mean(data)}")
        print(f"Standard deviation: {np.std(data)}")
    else:
        print(f"Dataset {dataset_path} not found in the file.")