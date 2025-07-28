import torch

class TrainingConfig:
    def __init__(self):
        # Environment settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # PPO settings
        self.total_timesteps = 3_000_000
        self.initial_lr = 3e-4
        self.decay_rate = 0.999
        self.min_lr = 1e-5
        self.ent_coef = 0.01
        self.vf_coef = 1.0

        # Logging & Checkpointing
        self.log_dir = './logs'
        self.tensorboard_log = './runs/tresette'
        self.checkpoint_path = './checkpoints/'
        self.clone_model_path = 'tresette_agent_clone'
        self.final_model_path = 'tresette_agent_final'

        # Self-play parameters
        self.clone_interval = 30_000
        self.heuristic_cutoff = 30_000