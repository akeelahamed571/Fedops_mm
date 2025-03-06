
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import gc

def extract_hospital_name(idx_info):
    """Extract hospital name from the idxInfo column."""
    return idx_info.split("-")[0]

def batch_convert_to_memmap(output_file, data_list, dtype=np.float32, batch_size=100):
    """
    Convert a list of data (all with the same shape) to a memmap array in batches.
    This avoids creating one huge NumPy array in memory.
    """
    n = len(data_list)
    if n == 0:
        np.save(output_file, np.array([], dtype=dtype))
        return
    # Determine the shape from the first element.
    sample = np.array(data_list[0], dtype=dtype)
    final_shape = (n,) + sample.shape
    mm = np.memmap(output_file, dtype=dtype, mode='w+', shape=final_shape)
    for i in range(0, n, batch_size):
        batch = np.array(data_list[i:i+batch_size], dtype=dtype)
        mm[i:i+batch_size] = batch
    mm.flush()
    del mm
    gc.collect()
    print(f"Saved batch array to {output_file} with shape {final_shape}")

def filter_static_features(static_x):
    """
    Remove hospital names from static features.
    Convert to a NumPy array (dtype object) and remove any string entries.
    """
    static_x_array = np.array(static_x, dtype=object)
    # Remove any element that is a string (assumed to be a hospital name)
    hospital_names = set(x for x in static_x_array if isinstance(x, str))
    static_x_filtered = np.array([x for x in static_x_array if x not in hospital_names], dtype=np.float32)
    return static_x_filtered if static_x_filtered.size > 0 else None

def remove_non_numeric_from_sequence(seq_x):
    """
    Given a sequence (list of rows), remove columns that are non-numeric.
    Assumes that all rows have the same structure.
    """
    # Check for empty input
    if isinstance(seq_x, np.ndarray):
        if seq_x.size == 0:
            return seq_x
    else:
        if not seq_x or len(seq_x) == 0:
            return seq_x

    # Determine non-numeric indices from the first row.
    first_row = seq_x[0]
    non_numeric_indices = [j for j, val in enumerate(first_row) if isinstance(val, str)]
    if not non_numeric_indices:
        return seq_x
    # Remove those columns from every row.
    cleaned = []
    for row in seq_x:
        cleaned_row = [val for j, val in enumerate(row) if j not in non_numeric_indices]
        cleaned.append(cleaned_row)
    return cleaned

# -----------------------------
# CONFIGURATION & OUTPUT PATHS
# -----------------------------
DATA_DIR = "processed_sequences_by_hospital"
ROOT_DIR = "HospitalsDataSplitting"
GLOBAL_DIR = os.path.join(ROOT_DIR, "GlobalData")
HOSPITALS_DIR = os.path.join(ROOT_DIR, "HospitalsData")
os.makedirs(GLOBAL_DIR, exist_ok=True)
os.makedirs(HOSPITALS_DIR, exist_ok=True)
print("Output directories created.")

# -----------------------------
# GLOBAL ACCUMULATORS
# -----------------------------
global_static_list = []
global_sequence_list = []
global_targets_list = []

# -----------------------------
# PROCESS EACH HOSPITAL FILE INDIVIDUALLY
# -----------------------------
# For each hospital, we split its sequences into a 10% global evaluation portion and 90% hospital-specific portion.
# Then we further split the hospital-specific portion into train/val/test.
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".pkl"):
        continue

    hospital_name = os.path.splitext(filename)[0]
    file_path = os.path.join(DATA_DIR, filename)
    print(f"\nProcessing hospital file: {file_path}")
    
    with open(file_path, "rb") as f:
        sequences = pickle.load(f)
    num_sequences = len(sequences)
    print(f"Loaded {num_sequences} sequences for hospital {hospital_name}")
    if num_sequences == 0:
        continue

    indices = np.arange(num_sequences)
    global_idx, hosp_idx = train_test_split(indices, test_size=0.9, random_state=42)
    print(f"Global portion: {len(global_idx)} sequences; Hospital-specific portion: {len(hosp_idx)} sequences")
    
    # Process global portion: apply filter and accumulate data.
    for i in global_idx:
        static_x, seq_x, target = sequences[i]
        filtered_static = filter_static_features(static_x)
        if filtered_static is None:
            continue
        seq_x_clean = remove_non_numeric_from_sequence(seq_x)
        try:
            seq_arr = np.array(seq_x_clean, dtype=np.float32)
        except Exception as e:
            print(f"Error converting sequence to float for hospital {hospital_name}: {e}")
            continue
        global_static_list.append(filtered_static)
        global_sequence_list.append(seq_arr)
        global_targets_list.append(np.array(target, dtype=np.float32))
    
    # Process hospital-specific portion.
    hosp_static_list = []
    hosp_sequence_list = []
    hosp_targets_list = []
    for i in hosp_idx:
        static_x, seq_x, target = sequences[i]
        filtered_static = filter_static_features(static_x)
        if filtered_static is None:
            continue
        seq_x_clean = remove_non_numeric_from_sequence(seq_x)
        try:
            seq_arr = np.array(seq_x_clean, dtype=np.float32)
        except Exception as e:
            print(f"Error converting hospital sequence to float for hospital {hospital_name}: {e}")
            continue
        hosp_static_list.append(filtered_static)
        hosp_sequence_list.append(seq_arr)
        hosp_targets_list.append(np.array(target, dtype=np.float32))
    
    print(f"Hospital {hospital_name}: {len(hosp_static_list)} valid sequences after filtering.")
    hosp_out_dir = os.path.join(HOSPITALS_DIR, hospital_name)
    os.makedirs(hosp_out_dir, exist_ok=True)
    
    # Further split hospital-specific data into train/val/test based on list indices.
    n = len(hosp_static_list)
    if n > 1:
        train_idx, temp_idx = train_test_split(np.arange(n), test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    else:
        train_idx, val_idx, test_idx = [0], [], []
    
    train_static = [hosp_static_list[i] for i in train_idx]
    val_static   = [hosp_static_list[i] for i in val_idx] if val_idx.size > 0 else []
    test_static  = [hosp_static_list[i] for i in test_idx] if test_idx.size > 0 else []
    
    train_sequence = [hosp_sequence_list[i] for i in train_idx]
    val_sequence   = [hosp_sequence_list[i] for i in val_idx] if val_idx.size > 0 else []
    test_sequence  = [hosp_sequence_list[i] for i in test_idx] if test_idx.size > 0 else []
    
    train_targets = [hosp_targets_list[i] for i in train_idx]
    val_targets   = [hosp_targets_list[i] for i in val_idx] if val_idx.size > 0 else []
    test_targets  = [hosp_targets_list[i] for i in test_idx] if test_idx.size > 0 else []
    
    # Save arrays using batch processing.
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "static_train.npy"), train_static, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "static_val.npy"), val_static, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "static_test.npy"), test_static, dtype=np.float32, batch_size=100)
    
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "sequence_train.npy"), train_sequence, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "sequence_val.npy"), val_sequence, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "sequence_test.npy"), test_sequence, dtype=np.float32, batch_size=100)
    
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "targets_train.npy"), train_targets, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "targets_val.npy"), val_targets, dtype=np.float32, batch_size=100)
    batch_convert_to_memmap(os.path.join(hosp_out_dir, "targets_test.npy"), test_targets, dtype=np.float32, batch_size=100)
    
    print(f"Hospital {hospital_name}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    del sequences, hosp_static_list, hosp_sequence_list, hosp_targets_list
    gc.collect()

# -----------------------------
# SAVE GLOBAL EVALUATION DATA
# -----------------------------
print("\nProcessing global evaluation data...")
batch_convert_to_memmap(os.path.join(GLOBAL_DIR, "static_data.npy"), global_static_list, dtype=np.float32, batch_size=100)
batch_convert_to_memmap(os.path.join(GLOBAL_DIR, "sequence_data.npy"), global_sequence_list, dtype=np.float32, batch_size=100)
batch_convert_to_memmap(os.path.join(GLOBAL_DIR, "targets.npy"), global_targets_list, dtype=np.float32, batch_size=100)
print(f"Saved {len(global_static_list)} global evaluation sequences.")

del global_static_list, global_sequence_list, global_targets_list
gc.collect()

print("\nData partitioning completed successfully!")
