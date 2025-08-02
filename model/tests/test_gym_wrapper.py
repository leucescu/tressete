import unittest
import numpy as np
import wandb
from model.gym_wrapper import TresetteGymWrapper


class TestTresetteGymWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wandb.init(mode="disabled")

    def setUp(self):
        self.gym_env = TresetteGymWrapper(opponent_policy="heuristic")

    def check_final_parameters(self):
        self.assertTrue(self.gym_env.env.done)
        self.assertEqual(self.gym_env.env.deck.cards_left(), 0)
        self.assertEqual(len(self.gym_env.env.players[0].hand.cards), 0)
        self.assertEqual(len(self.gym_env.env.players[1].hand.cards), 0)

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 40)

        total_points = self.gym_env.env.players[0].num_pts + self.gym_env.env.players[1].num_pts
        self.assertAlmostEqual(total_points, 11.0)

    def test_reset_and_initial_observation(self):
        reset_result = self.gym_env.reset()
        obs_dict = reset_result[0]  # reset returns (obs_dict, info)
        obs = obs_dict['obs']
        mask = obs_dict['action_mask']

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (214,))
        self.assertEqual(mask.shape, (10,))

        valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
        self.assertGreaterEqual(len(valid_actions), 1)

    def test_step_agent_and_opponent(self):
        self.gym_env.reset()
        done = False
        while not done:
            valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = self.gym_env.step(action)

            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

    def test_single_round_both_players_act(self):
        self.gym_env.reset()
        valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = self.gym_env.step(action)

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 2)

    def test_multiple_round_both_players_act(self):
        self.gym_env.reset()
        for _ in range(5):
            valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = self.gym_env.step(action)

        self.assertFalse(self.gym_env.env.done)
        self.assertEqual(self.gym_env.env.deck.cards_left(), 10)
        self.assertEqual(len(self.gym_env.env.players[0].hand.cards), 10)
        self.assertEqual(len(self.gym_env.env.players[1].hand.cards), 10)

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 10)

    def test_invalid_action_handling(self):
        self.gym_env.reset()
        invalid_action = 9999  # definitely invalid
        obs, reward, done, truncated, info = self.gym_env.step(invalid_action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)

    def test_game_over_after_20_rounds(self):
        self.gym_env.reset()
        done = False
        while not done:
            valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = self.gym_env.step(action)

        self.check_final_parameters()

    def test_play_5_games_in_a_row(self):
        for game in range(5):
            self.gym_env.reset()
            done = False
            while not done:
                valid_actions = np.where(self.gym_env.get_valid_action_mask())[0]
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = self.gym_env.step(action)

            self.check_final_parameters()
            self.assertTrue(done, f"Game {game + 1} did not finish properly")

    def test_action_mask_matches_valid_actions(self):
        env = TresetteGymWrapper(opponent_policy="heuristic")
        reset_result = env.reset()
        obs_dict = reset_result[0]
        obs = obs_dict['obs']
        mask = obs_dict['action_mask']

        self.assertEqual(obs.shape, (214,))
        self.assertEqual(mask.shape, (10,))

        # Validate binary mask
        self.assertTrue(np.all(np.isin(mask, [0.0, 1.0])), f"Mask contains non-binary values: {mask}")

        # Ensure some actions are valid
        self.assertGreaterEqual(np.sum(mask), 1, "No valid actions in mask")

        # Check alignment with environment's valid actions
        valid_actions = np.where(env.get_valid_action_mask())[0]
        expected_mask = np.zeros(10)
        expected_mask[valid_actions] = 1.0
        self.assertTrue(np.array_equal(mask, expected_mask),
                        f"Mismatch between env.get_valid_action_mask() and mask.\nExpected: {expected_mask}\nMask: {mask}")

class TestInvalidActions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wandb.init(mode="disabled")

    def setUp(self):
        self.env = TresetteGymWrapper(opponent_policy="heuristic")
        self.env.reset()

    def test_step_with_invalid_action_raises_error(self):
        invalid_action = 9999  # definitely invalid

        with self.assertRaises(ValueError) as cm:
            self.env.step(invalid_action)
        print("Caught expected error:", cm.exception)

    def test_step_with_invalid_action_returns_done(self):
        # If your environment does NOT raise but returns done, check that behavior:
        invalid_action = 9999
        try:
            result = self.env.step(invalid_action)
            print("Step output with invalid action:", result)
            # Check that done is True or False explicitly here if your env returns (obs, reward, done, info)
        except Exception as e:
            self.fail(f"step() raised unexpected exception: {e}")

    def test_valid_action_mask_consistency(self):
        mask = self.env.get_valid_action_mask()
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(np.any(mask), "No valid actions in mask")

    def test_all_actions_in_mask_are_valid(self):
        mask = self.env.get_valid_action_mask()
        valid_actions = np.where(mask)[0]

        for action in valid_actions:
            try:
                self.env.step(action)
            except Exception as e:
                self.fail(f"Valid action {action} raised exception: {e}")

if __name__ == "__main__":
    unittest.main()
