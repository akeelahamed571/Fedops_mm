# client_main.py
import sys
import json
import os
import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from omegaconf import DictConfig, OmegaConf
from fedops.client import client_utils
from fedops.client.app import FLClientTask
from models import TFTDataset, train_torch, test_torch  # from models.py

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Set random seeds for reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    print(OmegaConf.to_yaml(cfg))
    
    # Get the hospital index from the config override
    hospital_index = cfg.client.hospital_index
    
    # Load hospital mapping from hospital_mapping.json
    mapping_file = "hospital_mapping.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        hospital_mapping = json.load(f)
    
    hospital_name = None
    for name, idx in hospital_mapping.items():
        if idx == hospital_index:
            hospital_name = name
            break
    if hospital_name is None:
        print(f"Invalid hospital index: {hospital_index}")
        exit(1)
    print(f"Using data for hospital: {hospital_name}")
    
    
    # Construct path to hospital-specific data folder
    HOSPITAL_DATA_DIR = os.path.join("../../Safer/TFT/New_DP_TFT/HospitalsDataSplitting", "HospitalsData", hospital_name)
    
    # Define shapes for the npy files
    STATIC_SHAPE = (14,)        # 14 static features
    
    SEQUENCE_SHAPE = (192, 25)    # 192 time steps, 25 features per step
    TARGETS_SHAPE = (3,)          # 3 target variables
    
    # Load hospital-specific data using np.memmap (from data_preparation.py)
    from data_preparation import load_memmap_data
    static_train = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "static_train.npy"), STATIC_SHAPE)
    static_val   = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "static_val.npy"), STATIC_SHAPE)
    static_test  = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "static_test.npy"), STATIC_SHAPE)
    
    seq_train = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "sequence_train.npy"), SEQUENCE_SHAPE)
    seq_val   = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "sequence_val.npy"), SEQUENCE_SHAPE)
    seq_test  = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "sequence_test.npy"), SEQUENCE_SHAPE)
    
    targets_train = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "targets_train.npy"), TARGETS_SHAPE)
    targets_val   = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "targets_val.npy"), TARGETS_SHAPE)
    targets_test  = load_memmap_data(os.path.join(HOSPITAL_DATA_DIR, "targets_test.npy"), TARGETS_SHAPE)
    
    # Combine the arrays into lists of tuples for TFTDataset
    train_data = list(zip(static_train, seq_train, targets_train))
    val_data   = list(zip(static_val, seq_val, targets_val))
    test_data  = list(zip(static_test, seq_test, targets_test))
    
    # Create Dataset objects using TFTDataset defined in models.py
    train_dataset = TFTDataset(train_data)
    val_dataset = TFTDataset(val_data)
    test_dataset = TFTDataset(test_data)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    
    # Instantiate the multimodal model via Hydra config
    model = instantiate(cfg.model)
    
    # Check for an existing local model and download if available
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        model = client_utils.download_local_model(model_type=cfg.model_type,
                                                  task_id=task_id,
                                                  listdir=local_list,
                                                  model=model)
    
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": type(model).__name__,
        "train_torch": train_torch(),
        "test_torch": test_torch()
    }
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()

if __name__ == "__main__":
    main()
