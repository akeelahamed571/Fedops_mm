# server_main.py
import hydra
from omegaconf import DictConfig
from fedops.server.app import FLServer
from hydra.utils import instantiate
import models  # Contains TFTPredictor and test_torch
import data_preparation

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    # Use the global memmap data loader
    gl_val_loader = data_preparation.load_global_data(cfg.batch_size)
    
    fl_server = FLServer(cfg=cfg,
                         model=model,
                         model_name=type(model).__name__,
                         model_type=cfg.model_type,
                         gl_val_loader=gl_val_loader,
                         test_torch=models.test_torch())
    fl_server.start()

if __name__ == "__main__":
    main()
