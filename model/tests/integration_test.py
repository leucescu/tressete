import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import unittest

# Import your actual classes
from env.tresette_engine import TresetteEngine
from model.gym_wrapper import TresetteGymWrapper
from model.maskable_policy import MaskablePPOPolicy

class DummyActor(nn.Module):
    def __init__(self, input_dim=214, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, state, mask):
        logits = self.net(state)
        # Apply mask: mask is bool tensor shape (batch, actions)
        # For masked actions, put very negative logits
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask, logits, neg_inf * torch.ones_like(logits))
        return masked_logits

class DummyCritic(nn.Module):
    def __init__(self, input_dim=214):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

def dist_fn(logits):
    return Categorical(logits=logits)

class TestValidActionsIntegration(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine()
        self.wrapper = TresetteGymWrapper(self.env)

        actor = DummyActor()
        critic = DummyCritic()
        optim_ = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)

        self.policy = MaskablePPOPolicy(actor, critic, optim_, dist_fn)
        self.policy.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        self.env.reset()
        self.wrapper.reset()

    def test_valid_action_mask_and_policy(self):
        for step_idx in range(50):
            valid_actions = self.env.get_valid_actions()
            mask = self.wrapper.get_valid_action_mask()
            
            # Get current hand size for action validation
            player = self.env.players[self.env.current_player]
            hand_size = len(player.hand.cards)

            # Check 1: mask matches valid actions
            mask_indices = np.where(mask)[0]
            self.assertEqual(set(valid_actions), set(mask_indices),
                f"Mismatch between valid actions and mask at step {step_idx}")

            # Check 2: Not all actions valid at once (after step 0)
            if step_idx > 0:
                self.assertFalse(np.all(mask), f"All actions valid at step {step_idx}, unexpected")

            # Create dummy state tensor
            dummy_state = torch.randn(1, 214, device=self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits = self.policy.actor(dummy_state, mask_tensor)

                # Check 3: invalid logits masked (very negative)
                invalid_logits = logits[0, ~mask_tensor[0]]
                self.assertTrue(torch.all(invalid_logits < -1e9),
                    f"Invalid logits not masked correctly at step {step_idx}")

                dist = torch.distributions.Categorical(logits=logits)
                sampled_action = dist.sample().item()

            # Check 4: Sampled action is valid
            self.assertIn(sampled_action, valid_actions,
                f"Sampled invalid action {sampled_action} at step {step_idx}")

            # Step env with sampled valid action - expect only 2 returns
            self.env.step(sampled_action)

            # Check 5: Stepping invalid action raises ValueError
            # Find an invalid action considering hand size
            invalid_action = hand_size  # Always invalid since hand indices are 0 to hand_size-1
            with self.assertRaises(ValueError,
                msg=f"Invalid action {invalid_action} did not raise ValueError"):
                self.env.step(invalid_action)


class TestStateProgression(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine()
        self.wrapper = TresetteGymWrapper(self.env)
        self.env.reset()

    def test_state_changes_over_steps(self):
        prev_valid_actions = set(self.env.get_valid_actions())
        prev_mask = self.wrapper.get_valid_action_mask()

        for step_idx in range(1, 20):
            # Take a random valid action (to progress state)
            valid_actions = self.env.get_valid_actions()
            action = np.random.choice(valid_actions)
            self.env.step(action)  # Only returns 2 values
            
            # Get new valid actions and mask
            current_valid_actions = set(self.env.get_valid_actions())
            current_mask = self.wrapper.get_valid_action_mask()

            # Convert mask to indices
            mask_indices = set(np.where(current_mask)[0])

            # Check that the state actually changes: valid actions and mask should not be identical to previous step always
            self.assertFalse(current_valid_actions == prev_valid_actions and set(prev_mask) == mask_indices,
                             f"State did not change after step {step_idx}")

            # Check mask matches valid actions
            self.assertEqual(current_valid_actions, mask_indices,
                             f"Mismatch between valid actions and mask at step {step_idx}")

            # Update previous for next iteration
            prev_valid_actions = current_valid_actions
            prev_mask = current_mask


class TestEnvironmentBasics(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine()
        self.wrapper = TresetteGymWrapper(self.env)

    def test_reset_returns_valid_state(self):
        state = self.env.reset()
        self.assertIsNotNone(state, "Reset state is None")
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(len(valid_actions) > 0, "No valid actions after reset")
        mask = self.wrapper.get_valid_action_mask()
        self.assertEqual(set(valid_actions), set(np.where(mask)[0]),
                         "Valid actions mismatch mask after reset")

    def test_step_returns_expected_output(self):
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        action = valid_actions[0]

        # Expect only 2 returns (obs, done)
        obs, done = self.env.step(action)
        self.assertIsNotNone(obs, "Observation is None after step")
        self.assertIsInstance(done, bool, "Done flag is not boolean")

    def test_invalid_action_raises(self):
        self.env.reset()
        player = self.env.players[self.env.current_player]
        hand_size = len(player.hand.cards)
        invalid_action = hand_size  # Always invalid
        with self.assertRaises(ValueError):
            self.env.step(invalid_action)


class TestTrainingLoopStep(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine()
        self.wrapper = TresetteGymWrapper(self.env)
        actor = DummyActor()
        critic = DummyCritic()
        optim_ = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        self.policy = MaskablePPOPolicy(actor, critic, optim_, dist_fn)
        self.policy.to(torch.device("cpu"))
        self.env.reset()

    def test_single_training_step(self):
        valid_actions = self.env.get_valid_actions()
        state = torch.randn(1, 214)
        mask = torch.tensor(self.wrapper.get_valid_action_mask(), dtype=torch.bool).unsqueeze(0)

        logits = self.policy.actor(state, mask)
        dist = torch.distributions.Categorical(logits=logits)
        sampled_action = dist.sample().item()
        self.assertIn(sampled_action, valid_actions, "Sampled action not valid")

        # Step env - expect only 2 returns
        obs, done = self.env.step(sampled_action)
        self.assertIsInstance(done, bool)


class TestReproducibility(unittest.TestCase):
    def test_seed_reproducibility(self):
        seed = 12345
        torch.manual_seed(seed)
        np.random.seed(seed)

        env1 = TresetteEngine()
        env1.reset()
        acts1 = []
        for _ in range(10):
            valid = env1.get_valid_actions()
            action = np.random.choice(valid)
            env1.step(action)
            acts1.append(action)

        torch.manual_seed(seed)
        np.random.seed(seed)

        env2 = TresetteEngine()
        env2.reset()
        acts2 = []
        for _ in range(10):
            valid = env2.get_valid_actions()
            action = np.random.choice(valid)
            env2.step(action)
            acts2.append(action)

        # Convert numpy ints to Python ints for comparison
        acts1 = [int(a) for a in acts1]
        acts2 = [int(a) for a in acts2]
        self.assertEqual(acts1, acts2, "Actions differ across seeded runs")


class TestActionValidityAndIndexing(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine()
        self.env.reset()

    def test_valid_actions_are_within_hand_indices(self):
        for _ in range(20):
            valid_actions = self.env.get_valid_actions()
            player = self.env.players[self.env.current_player]
            hand_size = len(player.hand.cards)
            
            for action in valid_actions:
                self.assertTrue(0 <= action < hand_size,
                    f"Valid action {action} out of hand range [0, {hand_size})")
            
            action = np.random.choice(valid_actions)
            self.env.step(action)

    def test_step_rejects_out_of_range_actions(self):
        player = self.env.players[self.env.current_player]
        hand_size = len(player.hand.cards)
        invalid_actions = [-1, hand_size, hand_size + 1, 100]

        for action in invalid_actions:
            with self.assertRaises(ValueError):
                self.env.step(action)

    def test_valid_actions_correspond_to_hand_cards(self):
        for _ in range(10):
            valid_actions = self.env.get_valid_actions()
            player = self.env.players[self.env.current_player]
            hand_cards = player.hand.cards
            
            for idx in valid_actions:
                self.assertTrue(idx < len(hand_cards),
                    f"Action index {idx} exceeds hand size {len(hand_cards)}")
                card = hand_cards[idx]
                self.assertIsNotNone(card, "Card at action index is None")
            
            action = np.random.choice(valid_actions)
            self.env.step(action)

    def test_consistency_of_valid_actions_after_step(self):
        self.env.reset()
        prev_valid_actions = set(self.env.get_valid_actions())

        for _ in range(20):
            action = np.random.choice(list(prev_valid_actions))
            self.env.step(action)
            current_valid_actions = set(self.env.get_valid_actions())

            # Valid actions should never be empty until game done
            self.assertTrue(len(current_valid_actions) > 0 or self.env.done,
                "No valid actions available but game not done")

            prev_valid_actions = current_valid_actions


if __name__ == '__main__':
    unittest.main()
