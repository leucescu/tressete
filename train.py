import torch
import wandb

print("PyTorch version:", torch.__version__)
wandb.init(project="tresette-rl", mode="disabled")  # Later: mode="online"
print("Setup complete.")