import unittest
from env.tresette_env import Player, Card
from env.baseline_policies import (
    RandomPolicy,
    HighestLeadSuitPolicy,
    SimpleHeuristicPolicy
)

def make_hand(card_tuples):
    return [Card(suit, rank) for suit, rank in card_tuples]

class TestBaselinePolicies(unittest.TestCase):

    def setUp(self):
        self.player = Player(player_id=0)

    def test_random_policy_valid_index(self):
        self.player.receive_cards(make_hand([("bastoni", "1"), ("coppe", "3")]))
        policy = RandomPolicy()
        index = policy.get_action_index(self.player, lead_suit=None)
        self.assertIn(index, [0, 1])

    def test_highest_lead_suit_policy_selects_highest(self):
        self.player.receive_cards(make_hand([
            ("bastoni", "4"),
            ("bastoni", "3"),
            ("coppe", "1")
        ]))
        policy = HighestLeadSuitPolicy()
        index = policy.get_action_index(self.player, lead_suit="bastoni")
        self.assertEqual(self.player.hand[index].rank, "3")
        self.assertEqual(self.player.hand[index].suit, "bastoni")

    def test_highest_lead_suit_policy_random_fallback(self):
        self.player.receive_cards(make_hand([
            ("coppe", "2"),
            ("coppe", "5"),
            ("denari", "1")
        ]))
        policy = HighestLeadSuitPolicy()
        index = policy.get_action_index(self.player, lead_suit="spade")
        self.assertIn(index, [0, 1, 2])

    def test_heuristic_policy_leading_plays_highest(self):
        self.player.receive_cards(make_hand([
            ("spade", "4"),
            ("spade", "2"),
            ("denari", "3")
        ]))
        policy = SimpleHeuristicPolicy()
        index = policy.get_action_index(self.player, lead_suit=None)
        self.assertEqual(self.player.hand[index].rank, "3")
        self.assertEqual(self.player.hand[index].suit, "denari")

    def test_heuristic_policy_beats_card_if_possible(self):
        self.player.receive_cards(make_hand([
            ("spade", "2"),  # val 9
            ("spade", "5"),  # val 2
            ("denari", "1")  # val 8
        ]))
        trick = [(1, Card("spade", "1"))]  # val 8
        policy = SimpleHeuristicPolicy()
        index = policy.get_action_index(self.player, lead_suit="spade", trick=trick)
        self.assertEqual(self.player.hand[index].rank, "2")
        self.assertEqual(self.player.hand[index].suit, "spade")

    def test_heuristic_policy_plays_lowest_if_cannot_win(self):
        self.player.receive_cards(make_hand([
            ("spade", "5"),
            ("spade", "4")
        ]))
        trick = [(1, Card("spade", "3"))]  # val 10
        policy = SimpleHeuristicPolicy()
        index = policy.get_action_index(self.player, lead_suit="spade", trick=trick)
        self.assertEqual(self.player.hand[index].rank, "4")  # lower value
        self.assertEqual(self.player.hand[index].suit, "spade")

    def test_heuristic_policy_plays_lowest_of_other_suit_if_cannot_win(self):
        self.player.receive_cards(make_hand([
            ("denari", "5"),
            ("denari", "4")
        ]))
        trick = [(1, Card("spade", "3"))]  # val 10
        policy = SimpleHeuristicPolicy()
        index = policy.get_action_index(self.player, lead_suit="spade", trick=trick)
        self.assertEqual(self.player.hand[index].rank, "4")  # lower value
        self.assertEqual(self.player.hand[index].suit, "denari")


if __name__ == '__main__':
    unittest.main()
