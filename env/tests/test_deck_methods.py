import unittest
from env.tresette_engine import Card, Deck    

class TestDeck(unittest.TestCase):
    def test_deck_initialization(self):
            deck = Deck()
            self.assertEqual(len(deck.cards), 40)

    def test_deck_shuffle_changes_order(self):
        deck1 = Deck()
        deck2 = Deck()
        deck2.shuffle()
        self.assertNotEqual([str(c) for c in deck1.cards], [str(c) for c in deck2.cards])

    def test_draw_reduces_size(self):
        deck = Deck()
        start_len = len(deck.cards)
        card = deck.draw()
        self.assertIsInstance(card, Card)
        self.assertEqual(len(deck.cards), start_len - 1)

if __name__ == "__main__":
    unittest.main()