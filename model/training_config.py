import torch

class TrainingConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = 214  # as per your encoder output size
        self.action_dim = 10  # number of actions in your environment
        
        # Smaller hidden dims may help with stability, but 512 is okay if you have enough data
        self.hidden_dim = 256  
        self.actor_feature_dim = 128
        
        # Lower learning rate to reduce training instability
        self.lr = 3e-4  
        
        self.discount_factor = 0.99
        
        # Gradient clipping max norm (keep the same or slightly stricter)
        self.max_grad_norm = 0.5  
        
        self.estimation_step = 3
        self.num_train_envs = 8
        self.num_test_envs = 8
        
        # Increase epochs to allow more stable learning over smaller batches
        self.max_epoch = 100  
        
        # Fewer steps per epoch but more epochs to avoid sudden big jumps
        self.step_per_epoch = 500  
        
        # Repeat per collect: you can reduce this to lower variance in updates
        self.repeat_per_collect = 2  
        
        self.episode_per_test = 10
        
        self.batch_size = 64  # could increase to 128 if memory allows
        
        self.step_per_collect = 100  # collect less data per update for smoother learning
        
        self.initial_collect_step = 2000
        
        # Bigger buffer for more diverse training data, avoiding overfitting on small buffer
        self.buffer_size = 10000  
        
        self.max_total_steps = 5_000_000
        
        self.clone_interval = 1_000_000
        
        self.cutoff_steps = 150_000

        self.test_interval = 5000  # Test every 1000 steps

        # Regularization to prevent overfitting

        # Improves generalization by penalizing large weights

        # Helps stabilize training against noisy gradients
        self.weight_decay = 1e-4
        
        # Curriculum learning stages
        self.curriculum_stages = {
            0: 0,               # Stage 0: RandomPolicy (0-10k steps)
            5000: 1,           # Stage 1: HighestLeadSuitPolicy (10k-20k steps)
            15000: 2,           # Stage 2: SimpleHeuristicPolicy (20k-30k steps)
            30000: 3,           # Stage 3: AdvancedHeuristicPolicy (30k+ steps)
            self.cutoff_steps: 4 # Stage 4: Self-play
        }
