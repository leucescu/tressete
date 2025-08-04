import torch

class TrainingConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_train_envs = 8
        self.num_test_envs = 4
        self.buffer_size = 20000
        
        # Training parameters
        self.lr = 3e-4
        self.weight_decay = 1e-5
        self.discount_factor = 0.99
        self.max_grad_norm = 0.5
        self.vf_coef = 0.3
        self.eps_clip = 0.2
        self.action_dim = 10
        self.input_dim = 214
        self.hidden_dim = 128
        
        # Curriculum parameters
        self.random_win_rate = 0.85
        self.highest_lead_win_rate = 0.80
        self.simple_heuristic_win_rate = 0.75
        self.advanced_heuristic_win_rate = 0.70
        self.min_steps_per_stage = [5000, 10000, 15000, 20000]
        self.min_stage_duration = 20000  # Minimum steps per stage
        self.eval_episodes = 50
        self.clone_interval = 15000
        
        # Collection parameters
        self.max_total_steps = 3000000
        self.initial_collect_step = 2000
        self.step_per_collect = 100
        self.repeat_per_collect = 4
        self.batch_size = 512
        self.mini_batch_size = 128
        self.min_update_samples = 256
        self.test_interval = 5000
        self.episode_per_test = 20
        self.save_interval = 10000
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Learning rate schedule
        self.lr_schedule = {
            'milestones': [10000, 50000, 100000],
            'gamma': 0.5
        }