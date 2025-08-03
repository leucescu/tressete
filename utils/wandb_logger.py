import numpy as np
import wandb

class WandbLogger:
    def __init__(self, project_name="tressete-training", config=None):
        self.project_name = project_name
        self.config = config

    def log_train_data(self, data: dict, step: int):
        log_data = {f"train/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def log_test_data(self, data: dict, step: int):
        log_data = {f"test/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def log_update_data(self, data: dict, step: int):
        log_data = {f"update/{k}": float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else v
                    for k, v in data.items()}
        wandb.log(log_data, step=step)

    def save_data(self, data, name, step=None, env_step=None, gradient_step=None):
        pass