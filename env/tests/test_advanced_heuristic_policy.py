import unittest
from env.tresette_engine import TresetteEngine
from env.card import Card
from env.baseline_policies import AdvancedHeuristicPolicy

class TestAdvancedHeuristicPolicyIntegration(unittest.TestCase):

    def setUp(self):
        self.env = TresetteEngine(num_players=2, initial_hand_size=0)
        self.env.reset()
        self.p0 = self.env.players[0]
        self.p1 = self.env.players[1]
        self.env.current_player = 0

    def set_hand(self, player, card_list):
        player.hand.cards = card_list
        player.known_cards = card_list.copy()

    def test_safe_ace_play(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('spade', '1')])
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertEqual(action, 0, "Should safely play the Ace")

    def test_play_weakest_when_no_safe(self):
        self.env.reset()
        cards = [Card('spade', '5'), Card('coppe', '4'), Card('bastoni', '6')]
        self.set_hand(self.p0, cards)
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertIn(action, [0, 1, 2], "Should play one of the lowest cards")

    def test_avoid_unplayed_ace_suit(self):
        self.env.reset()
        cards = [Card('coppe', '5'), Card('spade', '3'), Card('bastoni', '7')]
        self.set_hand(self.p0, cards)
        self.p0.won_cards = [(0, Card('denari', '1')), (0, Card('denari', '2'))]
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertIn(action, [0, 2], "Should avoid playing suit with unplayed Ace")

    def test_second_play_and_can_beat(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('spade', '3'), Card('spade', '1')])
        self.env.trick.played_cards = [(1, Card('spade', '2'))]
        self.env.trick.lead_suit = 'spade'
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertEqual(action, 0, "Should beat with '3' not waste Ace")

    def test_second_play_and_cannot_beat(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('spade', '1'), Card('spade', '4')])
        self.env.trick.played_cards = [(1, Card('spade', '3'))]
        self.env.trick.lead_suit = 'spade'
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertEqual(action, 1, "Should not waste Ace if cannot win")

    def test_tied_points_prefers_lower_rank(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('denari', '13'), Card('coppe', '12')])
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertIn(action, [0, 1], "Should prefer lower rank or suit strategy")

    def test_first_player_plays_safest_card(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('coppe', '3'), Card('coppe', '1'), Card('denari', '7')])
        self.p0.won_cards = [(0, Card('coppe', '1'))]  # Ace already played
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertIn(action, [0, 2], "Should avoid exposing a new Ace")

    def test_play_when_only_one_valid_move(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('spade', '5')])
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertEqual(action, 0, "Should play the only card in hand")

    def test_leading_with_ace_if_no_risk(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('coppe', '1')])
        self.p0.won_cards = [(0, Card('coppe', '3')), (0, Card('coppe', '2'))]
        self.p1.won_cards = [(1, Card('coppe', '13'))]
        self.p1.known_cards = [Card('bastoni', '7')]
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertEqual(action, 0, "Should lead Ace safely when no threats remain")

    def test_fallback_to_weakest_when_uncertain(self):
        self.env.reset()
        self.set_hand(self.p0, [Card('coppe', '6'), Card('spade', '6'), Card('bastoni', '6')])
        action = AdvancedHeuristicPolicy.get_action_index(self.env)
        self.assertIn(action, [0, 1, 2], "Should choose the weakest card due to uncertainty")


class TestTresetteEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.env = TresetteEngine(num_players=2, initial_hand_size=10)
    def test_game_plays_to_completion(self):
        self.env.reset()
        self.assertFalse(self.env.done)
        self.assertEqual(len(self.env.players[0].hand.cards), 10)

        # Play the game until done
        while not self.env.done:
            action = AdvancedHeuristicPolicy.get_action_index(self.env)
            obs, done = self.env.step(action)

        # Final assertions
        self.assertTrue(self.env.done)
        self.assertEqual(len(self.env.deck.cards), 0)
        self.assertEqual(len(self.env.players[0].hand.cards), 0)
        self.assertEqual(len(self.env.players[1].hand.cards), 0)

        total_cards_won = len(self.env.players[0].won_cards) + len(self.env.players[1].won_cards)
        self.assertEqual(total_cards_won, 40, "All 40 cards should be won after the game")

        total_points = self.env.players[0].num_pts + self.env.players[1].num_pts
        self.assertAlmostEqual(total_points, 11.0, msg="Total points after bonus should be 11.0")


if __name__ == '__main__':
    unittest.main()
