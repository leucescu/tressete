from env.deck import Deck
from env.player import Player
from env.card import Card
from env.trick import Trick


class TresetteEngine:
    def __init__(self, num_players=2, initial_hand_size=10):
        self.num_players = num_players
        self.initial_hand_size = initial_hand_size
        self.deck = Deck()
        self.players = [Player(i) for i in range(num_players)]
        self.reset()

    def reset(self):
        self.deck = Deck()
        self.deck.shuffle()

        # Deal initial hands
        hands = self.deck.deal(self.num_players, self.initial_hand_size)
        for i, player in enumerate(self.players):
            player.reset()
            player.hand.receive_cards(hands[i])

        self.current_player = 0
        self.trick =  Trick()
        self.done = False

        # Revisit this
        self.final_trick_winner = None

        # Keep track of points per trick to avoid double counting
        self.points_accumulated = 0.0

        return self._get_obs()

    # Step performed for each players turn
    def step(self, action_idx: int):
        # if self.done:
        #     raise Exception("Game over. Call reset().")

        player = self.players[self.current_player]
        lead_suit = self.trick[0][1].suit if self.trick else None
        valid_actions = player.get_valid_moves(lead_suit)
        if action_idx not in valid_actions:
            raise ValueError(f"Invalid action {action_idx}. Valid actions: {valid_actions}")

        played_card = player.play_card(action_idx)
        self.trick.append((self.current_player, played_card))

        # Next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Trick complete
        if len(self.trick) == self.num_players:
            winner = self._resolve_trick()
            trick_cards = [card for _, card in self.trick]

            # Calculate trick points separately and add only once
            trick_points = sum(card.point_value for card in trick_cards)

            self.players[winner].collect_trick(trick_cards)
            self.points_accumulated += trick_points

            # Set last_trick_pts for winner for reward use
            for player in self.players:
                player.last_trick_pts = 0.0
            self.players[winner].last_trick_pts = trick_points

            self.trick = []
            self.current_player = winner

        # Check done condition
        hands_empty = all(len(p.hand) == 0 for p in self.players)
        deck_empty = len(self.deck.cards) == 0
        self.done = hands_empty and deck_empty

        if self.done:
            # Calculate total points before bonus
            total_points = sum(p.num_pts for p in self.players)

            # Bonus points to assign so total points add to 11
            bonus_points = 11.0 - total_points

            # Defensive check (should rarely happen if logic is correct)
            if bonus_points < 0:
                bonus_points = 0.0

            self.players[winner].update_points_at_the_end(bonus_points)

            # Reset last trick pts since game ended
            for player in self.players:
                player.last_trick_pts = 0.0

        return self._get_obs(), self.done

    def step2(self, action_idx):
        valid_actions = self.get_valid_moves(lead_suit)
        if action_idx not in valid_actions:
            raise ValueError(f"Invalid action {action_idx}. Valid actions: {valid_actions}")


    # def get_valid_actions(self):
    #     player = self.players[self.current_player]
    #     lead_suit = self.trick[0][1].suit if self.trick else None
    #     return player.get_valid_moves(lead_suit)
    
    def _get_obs(self):
        current_hand = self.players[self.current_player].hand
        trick_cards = [card for _, card in self.trick]

        other_known_cards = []
        for player in self.players:
            other_known_cards.extend(player.known_cards)

        return {
            "hand": current_hand,
            "trick": trick_cards,
            "current_player": self.current_player,
            "cards_left_in_deck": len(self.deck.cards),
            "other_players_known_cards": other_known_cards
        }

