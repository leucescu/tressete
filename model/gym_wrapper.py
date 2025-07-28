import gym
import wandb
import numpy as np
from gym import spaces
from env.tresette_engine import TresetteEngine
from model.state_encoder import encode_state
from env.baseline_policies import SimpleHeuristicPolicy, SlightlySmarterHeuristicPolicy

class TresetteGymWrapper(gym.Env):
    def __init__(self, opponent_model=None, opponent_policy=None, device='cpu'):
        super().__init__()
        self.env = TresetteEngine()
        self.agent_index = 0
        self.opponent_model = opponent_model
        self.opponent_policy = opponent_policy  # "heuristic", "random", or None
        self.device = device

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(204,), dtype=np.float32)
        self.episode_counter = 0

    def reset(self):
        self.env.reset()
        return encode_state(self.env._get_obs(), self.env.players[self.agent_index])

    def step(self, action):
        if self.env.current_player == self.agent_index:
            self.play_agents_action(action)
            if self.env.done:
                raise ValueError("Game ended before opponent's turn could be played.")
            self.play_opponents_action()
        else:
            self.play_opponents_action()
            if self.env.done:
                raise ValueError("Game ended before agent's turn could be played.")
            self.play_agents_action(action)

        reward = self.get_reward_value(self.env.done)

        return encode_state(self.env._get_obs(), self.env.players[self.agent_index]), reward, self.env.done, {}

    def get_valid_actions(self):
        return self.env.get_valid_actions()
    
    def get_reward_value(self, done):
        reward = self.env.players[self.agent_index].last_trick_pts

        if done:
            agent_points = self.env.players[self.agent_index].num_pts
            self.episode_counter += 1
            wandb.log({"agent_pts_per_game": agent_points}, step=self.episode_counter)

            # Final reward adjustment
            reward += (agent_points - 5.5)
            wandb.log({"reward": reward}, step=self.episode_counter)

        return reward / 7.5  # Normalize reward to [-1, 1] range

    def play_opponents_action(self):
        valid_actions = self.env.get_valid_actions()

        if self.opponent_policy == "heuristic":
            player = self.env.players[self.env.current_player]
            trick = self.env.trick
            lead_suit = trick.lead_suit if trick else None
            action_opponent = SlightlySmarterHeuristicPolicy.get_action_index(player, lead_suit, trick)
        elif self.opponent_model:
            obs_raw = self.env._get_obs()
            encoded = encode_state(obs_raw, self.env.players[self.env.current_player])
            action_opponent, _ = self.opponent_model.predict(encoded, deterministic=True)
            if action_opponent not in valid_actions:
                action_opponent = np.random.choice(valid_actions)
        else:
            action_opponent = np.random.choice(valid_actions)

        _, _ = self.env.step(action_opponent)
        
    def play_agents_action(self, action):
        valid_actions = self.env.get_valid_actions()

        if action not in valid_actions:
            # Uncomment for debugging:
            # print(f"[WARN] Invalid action {action}. Valid: {valid_actions}")
            action = np.random.choice(valid_actions)

        _, _ = self.env.step(action)