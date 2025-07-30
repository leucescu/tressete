import torch

class TrainingConfig:
    def __init__(self):
        # Environment settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # PPO settings
        self.total_timesteps = 5_000_000
        self.initial_lr = 2e-4
        self.decay_rate = 0.9995
        self.min_lr = 1e-5
        self.vf_coef = 1.5

        # Logging & Checkpointing
        self.log_dir = './logs'
        self.tensorboard_log = './runs/tresette'
        self.checkpoint_path = './checkpoints/'
        self.clone_model_path = 'tresette_agent_clone'
        self.final_model_path = 'tresette_agent_final'

        # Self-play parameters
        self.clone_interval = 150_000
        self.heuristic_cutoff = 150_000

        self.ent_coef_start = 0.15
        self.ent_coef_final = 0.001
        self.kl_coef = 0.015
        self.gamma=0.995          # Increase from default 0.99
        self.gae_lambda=0.985     # Higher lambda for longer credit assignment
        self.n_steps=1024 