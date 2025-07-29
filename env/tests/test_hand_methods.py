import unittest
from env.hand import Hand
from env.deck import Deck
from env.card import Card

class TestHand(unittest.TestCase):

    def setUp(self):
        self.hand = Hand()
        self.deck = Deck()
        self.deck.shuffle()

    def test_receive_cards_adds_cards_to_hand(self):
        initial_size = len(self.hand.cards)
        cards_to_receive = [self.deck.draw() for _ in range(3)]
        self.hand.receive_cards(cards_to_receive)
        
        self.assertEqual(len(self.hand.cards), initial_size + 3)
        for card in cards_to_receive:
            self.assertIn(card, self.hand.cards)

    def test_add_single_to_hand_increases_size(self):
        initial_size = len(self.hand.cards)
        card = self.deck.draw()
        self.hand.add_single_to_hand(card)
        
        self.assertEqual(len(self.hand.cards), initial_size + 1)
        self.assertIn(card, self.hand.cards)

    def test_suit_counts_correctly_count_cards_by_suit(self):
        cards = [Card('coppe', '3'), Card('spade', '1'), Card('coppe', '7')]
        self.hand.receive_cards(cards)
        
        counts = self.hand.suit_counts()
        self.assertEqual(counts['coppe'], 2)
        self.assertEqual(counts['spade'], 1)

    def test_most_common_suit(self):
        cards = [Card('coppe', '3'), Card('spade', '1'), Card('coppe', '7')]
        self.hand.receive_cards(cards)
        
        counts = self.hand.suit_counts()
        most_common_suit = max(counts.items(), key=lambda x: x[1])[0]
        self.assertEqual(most_common_suit, 'coppe')