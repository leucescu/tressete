import unittest
import numpy as np
import wandb
from model.gym_wrapper import TresetteGymWrapper

class TestTresetteGymWrapper(unittest.TestCase):
    def setUp(self):
        self.gym_env = TresetteGymWrapper()

    @classmethod
    def setUpClass(cls):
        wandb.init(mode="disabled") 

    def check_final_parameters(self):
        self.assertEqual(self.gym_env.env.done, True)
        self.assertEqual(self.gym_env.env.deck.cards_left(), 0)
        self.assertEqual(len(self.gym_env.env.players[0].hand.cards), 0)
        self.assertEqual(len(self.gym_env.env.players[1].hand.cards), 0)

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 40)

        total_points = self.gym_env.env.players[0].num_pts + self.gym_env.env.players[1].num_pts
        self.assertAlmostEqual(total_points, 11.0)

    def test_reset_and_initial_observation(self):
        obs = self.gym_env.env.reset()
        self.assertIsInstance(obs, dict, "Reset should return observation dict")
        valid_actions = self.gym_env.get_valid_actions()
        self.assertTrue(len(valid_actions) == 10, "There should be valid actions after reset")

    def test_step_agent_and_opponent(self):
        obs = self.gym_env.env.reset()
        done = False
        while not done:
            valid_actions = self.gym_env.get_valid_actions()
            action = np.random.choice(valid_actions)
            obs, reward, done, info = self.gym_env.step(action)
            self.assertIsInstance(obs, np.ndarray, "Observation should be encoded numpy array")
            self.assertIsInstance(reward, float, "Reward should be a float")
            self.assertIsInstance(done, bool, "Done flag should be boolean")

    def test_single_round_both_players_act(self):
        obs = self.gym_env.reset()
        valid_actions = self.gym_env.get_valid_actions()
        action = np.random.choice(valid_actions)

        # Step once — should include agent and opponent move
        obs, reward, done, info = self.gym_env.step(action)

        # Check trick state (both cards played)
        current_trick = self.gym_env.env.trick

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 2)

    def test_multiple_round_both_players_act(self):
        obs = self.gym_env.reset()
        for _ in range(5):
            valid_actions = self.gym_env.get_valid_actions()
            action = np.random.choice(valid_actions)

            # Step once — should include agent and opponent move
            obs, reward, done, info = self.gym_env.step(action)

            # Check trick state (both cards played)
            current_trick = self.gym_env.env.trick
            self.assertEqual(len(current_trick.played_cards), 0)

        # Check that state, reward, and done flag are as expected
        self.assertEqual(self.gym_env.env.done, False)
        self.assertEqual(self.gym_env.env.deck.cards_left(), 10)
        self.assertEqual(len(self.gym_env.env.players[0].hand.cards), 10)
        self.assertEqual(len(self.gym_env.env.players[1].hand.cards), 10)

        played_cards = len(self.gym_env.env.players[0].won_cards) + len(self.gym_env.env.players[1].won_cards)
        self.assertEqual(played_cards, 10)

    def test_invalid_action_handling(self):
        self.gym_env.env.reset()
        invalid_action = 9999  # definitely invalid
        obs, reward, done, info = self.gym_env.step(invalid_action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)

    def test_game_over_after_20_rounds(self):
        self.gym_env.env.reset()
        for _ in range(20):
            valid_actions = self.gym_env.get_valid_actions()
            action = np.random.choice(valid_actions)
            obs, reward, done, info = self.gym_env.step(action)

        # Check that state, reward, and done flag are as expected
        self.check_final_parameters()

    def test_play_5_games_in_a_row(self):
        self.gym_env.env.reset()
        for game in range(5):
            while not self.gym_env.env.done:
                valid_actions = self.gym_env.get_valid_actions()
                action = np.random.choice(valid_actions)
                obs, reward, done, info = self.gym_env.step(action)
                
            # After each game, verify done is True and the environment is consistent
            self.check_final_parameters()
            self.assertTrue(done, f"Game {game + 1} did not finish properly")

if __name__ == "__main__":
    unittest.main()
