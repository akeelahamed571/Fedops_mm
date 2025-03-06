# data_preparation.py
import os
import numpy as np
from torch.utils.data import DataLoader
from models import TFTDataset

# -------------------------------------------------------------------
# Helper: Efficient Memmap Data Loading
# -------------------------------------------------------------------
def load_memmap_data(file_path, sample_shape, dtype=np.float32):
    """
    Loads a memory-mapped NumPy array efficiently.
    """
    if not os.path.exists(file_path):
        return None
    file_size = os.path.getsize(file_path)
    bytes_per_sample = np.prod(sample_shape) * np.dtype(dtype).itemsize
    num_samples = file_size // bytes_per_sample
    return np.memmap(file_path, dtype=dtype, mode="r", shape=(num_samples,) + sample_shape)

# -------------------------------------------------------------------
# Global Data Loading (Efficiently with Memmap)
# -------------------------------------------------------------------
GLOBAL_DIR = "../../Safer/TFT/New_DP_TFT/HospitalsDataSplitting/GlobalData"

# Define shapes based on dataset properties
STATIC_SHAPE = (14,)        # Example: 14 static features
SEQUENCE_SHAPE = (192, 25)  # Example: 192 time steps, 25 features per time step
TARGETS_SHAPE = (3,)        # Example: 3 target variables

def load_global_data(batch_size):
    static_global = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"), STATIC_SHAPE)
    seq_global = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQUENCE_SHAPE)
    targets_global = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"), TARGETS_SHAPE)
    
    # Combine the arrays into a list of tuples for TFTDataset
    global_data = list(zip(static_global, seq_global, targets_global))
    
    global_dataset = TFTDataset(global_data)
    test_loader = DataLoader(global_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Optionally, if you also want to load hospital‐specific data with memmap,
# you can define a similar function in data_preparation.py.
