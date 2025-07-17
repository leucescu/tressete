import unittest
import random
from env.tresette_env import TresetteEnv, Card, Player, Deck


class TestTresetteEnv(unittest.TestCase):
    
    def setUp(self):
        self.env = TresetteEnv(num_players=2, initial_hand_size=10)

    def test_reset_environment(self):
        obs = self.env.reset()
        self.assertEqual(len(obs["hand"]), 10)
        self.assertEqual(obs["current_player"], 0)
        self.assertEqual(len(self.env.deck.cards), 20)

    def test_valid_actions(self):
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(len(valid_actions) == 10)
        self.assertTrue(all(isinstance(i, int) for i in valid_actions))

    def test_step_progression(self):
        self.env.reset()
        turns = 0
        while not self.env.done:
            valid_actions = self.env.get_valid_actions()
            action = random.choice(valid_actions)
            _, _, _, _ = self.env.step(action)
            turns += 1

        # After the game ends:
        self.assertTrue(self.env.done)
        total_cards_won = sum(len(p.won_cards) for p in self.env.players)
        self.assertEqual(total_cards_won, 40)
        self.assertEqual(turns, 40)

    def test_point_sum_reasonable(self):
        self.env.reset()
        while not self.env.done:
            valid_actions = self.env.get_valid_actions()
            action = random.choice(valid_actions)
            _, _, _, _ = self.env.step(action)

        total_points = sum(p.num_pts for p in self.env.players)
        self.assertAlmostEqual(total_points, 11)

    def test_that_the_number_of_cards_is_the_same_after_10_steps(self):
        self.env.reset()
        for _ in range(10):
            valid_actions = self.env.get_valid_actions()
            action = random.choice(valid_actions)
            _, _, _, _ = self.env.step(action)

        first_value = len(self.env.players[0].hand)
        for player in self.env.players:
            self.assertEqual(first_value, len(player.hand))

    def test_trick_resolution(self):
        self.env.reset()
        self.env.trick = [
            (0, Card('coppe', '3')),
            (1, Card('coppe', '1'))
        ]
        winner = self.env._resolve_trick()
        self.assertEqual(winner, 0)

    def test_trick_resolution_different_suit(self):
        self.env.reset()
        self.env.trick = [
            (0, Card('coppe', '1')),
            (1, Card('bastoni', '3'))
        ]
        winner = self.env._resolve_trick()
        self.assertEqual(winner, 0)

    def test_get_valid_moves_follow_suit(self):
        player = self.env.players[0]
        # Manually give player cards with mixed suits
        player.hand = [
            Card('coppe', '3'),
            Card('bastoni', '1'),
            Card('coppe', '7')
        ]
        valid_moves = player.get_valid_moves('coppe')
        # Should only be indices of coppe cards (0 and 2)
        self.assertEqual(valid_moves, [0, 2])

        valid_moves_no_lead = player.get_valid_moves()
        self.assertEqual(valid_moves_no_lead, [0, 1, 2])

    def test_invalid_move_raises(self):
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        invalid_action = max(valid_actions) + 1
        with self.assertRaises(ValueError):
            self.env.step(invalid_action)


class TestPlayerMethods(unittest.TestCase):

    def setUp(self):
        self.player = Player(player_id=0)
        self.deck = Deck()
        self.deck.shuffle()

    def test_draw_card_adds_card_to_hand_and_known(self):
        initial_hand_size = len(self.player.hand)
        initial_known_size = len(self.player.known_cards)
        
        card_drawn = self.player.draw_card(self.deck)
        
        self.assertIsInstance(card_drawn, Card)
        self.assertEqual(len(self.player.hand), initial_hand_size + 1)
        self.assertEqual(len(self.player.known_cards), initial_known_size + 1)
        self.assertIn(card_drawn, self.player.hand)
        self.assertIn(card_drawn, self.player.known_cards)

    def test_collect_trick_adds_cards_to_won_and_updates_points(self):
        cards = [Card('coppe', '3'), Card('spade', '1')]
        initial_won_cards = len(self.player.won_cards)
        initial_points = self.player.num_pts
        
        self.player.collect_trick(cards)
        
        self.assertEqual(len(self.player.won_cards), initial_won_cards + len(cards))
        for card in cards:
            self.assertIn(card, self.player.won_cards)
        
        expected_points = initial_points + sum(card.point_value for card in cards)
        self.assertAlmostEqual(self.player.num_pts, expected_points)

if __name__ == "__main__":
    unittest.main()