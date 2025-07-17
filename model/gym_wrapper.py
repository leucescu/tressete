import gym
import wandb
from gym import spaces
import numpy as np
from env.tresette_env import TresetteEnv
from model.state_encoder import encode_state

class TresetteGymWrapper(gym.Env):
    def __init__(self, opponent_model=None, device='cpu'):
        super().__init__()
        self.env = TresetteEnv()
        self.agent_index = 0  # main agent always player 0
        self.opponent_model = opponent_model
        self.device = device

        self.action_space = spaces.Discrete(10)  # max hand size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(162,), dtype=np.float32)

        self.episode_counter = 0

    def reset(self):
        obs = self.env.reset()
        return encode_state(obs, self.env.players[self.agent_index])

    def step(self, action):
        # Main agent acts (player 0)
        if action not in self.env.get_valid_actions():
            # Replace invalid action with a valid one (e.g., random valid action)
            valid_actions = self.env.get_valid_actions()
            action = valid_actions[0]  # or random.choice(valid_actions)
    
        obs, done = self.env.step(action)
        reward = self.env.players[self.agent_index].last_trick_pts

        # If done, add final reward based on game points
        if done:
            agent_points = self.env.players[0].num_pts  # or adapt path to your env
            self.episode_counter += 1
            wandb.log({"agent_pts_per_game": agent_points}, step=self.episode_counter)
            reward += (self.env.players[self.agent_index].num_pts - 5.5) * 5

        # Opponent acts if not done
        while not done and self.env.current_player != self.agent_index:
            valid_actions = self.env.get_valid_actions()
            if self.opponent_model is not None:
                # Prepare observation for opponent
                opp_obs_raw = self.env._get_obs()
                opp_obs_encoded = encode_state(opp_obs_raw, self.env.players[self.env.current_player])
                # Get action from opponent model
                action_opponent, _ = self.opponent_model.predict(opp_obs_encoded, deterministic=True)
                if action_opponent not in valid_actions:
                    # Mask invalid action by picking random valid action
                    action_opponent = np.random.choice(valid_actions)
            else:
                # If no opponent model, pick random valid action
                action_opponent = np.random.choice(valid_actions)

            obs, done = self.env.step(action_opponent)

        return encode_state(obs, self.env.players[self.agent_index]), reward, done, {}

    def get_valid_actions(self):
        return self.env.get_valid_actions()
