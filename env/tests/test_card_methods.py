import unittest
from env.tresette_engine import Card

class TestCard(unittest.TestCase):

    def setUp(self):
        return super().setUp()
    
    def test_card_equality(self):
        self.assertEqual(Card('spade', '3'), Card('spade', '3'))
        self.assertNotEqual(Card('spade', '3'), Card('coppe', '3'))

    def test_card_value_and_point(self):
        card = Card('denari', '3')
        self.assertEqual(card.value(), 10)
        self.assertAlmostEqual(card.point_value, 1/3)