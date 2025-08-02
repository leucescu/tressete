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
        
        self.max_total_steps = 200_000
        
        self.clone_interval = 1_000
        
        self.cutoff_steps = 1_000
