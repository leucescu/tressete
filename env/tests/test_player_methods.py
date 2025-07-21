import unittest
from env.tresette_engine import Card, Player, Deck


class TestPlayer(unittest.TestCase):

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