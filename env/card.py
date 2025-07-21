from env.card_config import POINT_PER_RANK, RANK_ORDER

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.point_value = POINT_PER_RANK[self.rank]

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

    def value(self):
        return RANK_ORDER[self.rank]