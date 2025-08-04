import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.tresette_engine import TresetteEngine
from model.state_encoder import EncodedState
from env.baseline_policies import AdvancedHeuristicPolicy, SimpleHeuristicPolicy, HighestLeadSuitPolicy, RandomPolicy


class TresetteGymWrapper(gym.Env):
    def __init__(self, opponent_model=None, opponent_policy="advanced_heuristic", device='cpu'):
        super().__init__()
        self.env = TresetteEngine()
        self.agent_index = 0
        self.opponent_model = opponent_model
        self.opponent_policy = opponent_policy
        self.device = device
        self.state_encoder = EncodedState()
        self.agents_reward_for_next_turn = 0.0
        self.step_count = 0

        # Action and observation space
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0.0, high=1.0, shape=(214,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.action_space.n)
        })

    def _get_obs(self):
        self.env.current_player
        if self.env.current_player == self.agent_index:
            encoded = self.state_encoder.agent_state
        else:
            encoded = self.state_encoder.opponent_state
        if np.isnan(encoded).any():
            print("WARNING: NaN in state encoding!")
            encoded = np.nan_to_num(encoded, nan=0.0)
        if not isinstance(encoded, np.ndarray) or encoded.shape != (214,):
            raise ValueError(f"Encoded state must be a numpy array of shape (214,), got {type(encoded)} with shape {getattr(encoded, 'shape', None)}")
        return encoded

    def reset(self):
        self.env.reset()
        self.state_encoder = EncodedState()
        self.state_encoder.update_player_state(self)
        obs = self._get_obs()
        self.agents_reward_for_next_turn = 0.0
        self.step_count = 0

        # Get mask from the encoded state of the current player
        if self.env.current_player == self.agent_index:
            mask = self.state_encoder.agent_state[-10:]
        else:
            mask = self.state_encoder.opponent_state[-10:]

        return {"obs": obs, "action_mask": mask}, {}

    def step(self, action):
        # responsible for invalid actions and non iterables
        if self.env.current_player == self.agent_index:
            self._play_agent_action(action)
        else:
            self._play_opponent_action()

        terminated = False
        truncated = self.step_count >= 40
        terminated = self.env.done or truncated
        reward = self._get_reward(self.env.done)

        self.state_encoder.update_player_state(self)
        # Return only the valid action mask of the current player
        if self.env.current_player == self.agent_index:
            mask = self.state_encoder.agent_state[-10:]
        else:
            mask = self.state_encoder.opponent_state[-10:]

        # Get the observation after the step
        obs = self._get_obs()
        if self.env.done:
            pass

        return {"obs": obs, "action_mask": mask}, reward, terminated, truncated, {}

    # When should agent get the reward if played first?
    def _get_reward(self, done):
        if len(self.env.trick.played_cards) != 0:
            return 0.0
        elif self.env.current_player != self.agent_index:
            if len(self.env.trick.played_cards) == 0:
                self.agents_reward_for_next_turn = self.env.players[self.agent_index].last_trick_pts
            return 0.0
        else:
            reward = self.env.players[self.agent_index].last_trick_pts + self.agents_reward_for_next_turn
            self.agents_reward_for_next_turn = 0.0
            if done:
                agent_pts = self.env.players[self.agent_index].num_pts
                # reward += (agent_pts - 5.5) * 1.5
                reward += (agent_pts - 7)
                # Add higher reward for great game, so far added only for advanced heuristic policy
                if self.opponent_policy == "advanced_heuristic":
                    if self.env.players[self.agent_index].num_pts > 8.0:
                        reward += 1
                    if self.env.players[self.agent_index].num_pts > 9.0:
                        reward += 2
                    if self.env.players[self.agent_index].num_pts > 10:
                        reward += 4
            
            return reward / 11.0 # Normalize reward to be between 0 and 1

    def _play_opponent_action(self):
        if self.opponent_policy == "random":
            action = RandomPolicy.get_action_index(self.env)
        elif self.opponent_policy == "highest_lead":
            action = HighestLeadSuitPolicy.get_action_index(self.env)
        elif self.opponent_policy == "simple_heuristic":
            action = SimpleHeuristicPolicy.get_action_index(self.env)
        elif self.opponent_policy == "advanced_heuristic":
            action = AdvancedHeuristicPolicy.get_action_index(self.env)
        elif self.opponent_model:
            obs = torch.tensor(self.state_encoder.opponent_state, dtype=torch.float32).unsqueeze(0)

            valid_actions = self.env.get_valid_actions()
            mask = torch.zeros(self.action_space.n, dtype=torch.bool)
            mask[valid_actions] = True

            opponent_model_action, _ = self.opponent_model.predict(obs, mask=mask, deterministic=True)
            action = opponent_model_action[0]
            if action not in valid_actions:
                raise ValueError(f"Invalid opponent action {action}. Valid actions: {valid_actions}")

        self.env.step(action)

    def _play_agent_action(self, action):
        valid = self.env.get_valid_actions()
        if action not in valid:
            print("Invalid agent action:", action)
            print("Valid actions:", valid)
            raise ValueError(f"Invalid action {action}. Valid actions: {valid}")
        self.env.step(action)
