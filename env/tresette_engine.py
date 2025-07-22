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

    # Step performed for each player's turn
    def step(self, action_idx: int):
        # if self.done:
        #     raise Exception("Game over. Call reset().")

        player = self.players[self.current_player]
        lead_suit = self.trick.lead_suit
        valid_actions = player.get_valid_moves(lead_suit)

        if action_idx not in valid_actions:
            raise ValueError(f"Invalid action {action_idx}. Valid actions: {valid_actions}")

        # Played card is added to te trick
        played_card = player.play_card(action_idx)
        self.trick.played_cards.append((self.current_player, played_card))

        # Set lead suit
        if self.trick.lead_suit is None:
            self.trick.lead_suit = self.trick.played_cards[0][1].suit

        # Next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Trick complete
        if len(self.trick.played_cards) == self.num_players:
            played_cards = self.trick.played_cards
            winner, points = self.trick.resolve_trick()

            self.players[winner].collect_trick(played_cards, points)

            # Player who won the trick gets to play next
            self.current_player = winner

            self.draw_cards(winner)

        # Check if the game is over
        self.done = self._is_game_over()

        if self.done:
            self.add_bonus_points(self.players[winner])


        return self._get_obs(), self.done

    def draw_cards(self, winner):
        # Winner draws first
        current_draw_index = winner

        for _ in range(self.num_players):
            self.players[current_draw_index].draw_card(self.deck)
            current_draw_index= (current_draw_index + 1) % self.num_players


    def _is_game_over(self):
        hands_empty = all(len(player.hand.cards) == 0 for player in self.players)
        deck_empty = len(self.deck.cards) == 0
        return hands_empty and deck_empty

    def add_bonus_points(self, player: Player):
        # Calculate total points before bonus
        total_points = sum(player.num_pts for player in self.players)

        # Bonus points to assign so total points add to 11
        bonus_points = 11.0 - total_points

        # Defensive check (should rarely happen if logic is correct)
        if bonus_points < 0:
            raise ValueError(f"Bonus points value: {bonus_points} is not valid.")

        player.update_points_at_the_end(bonus_points)

    def get_valid_actions(self):
        player = self.players[self.current_player]
        return player.get_valid_moves(self.trick.lead_suit)
    
    def _get_obs(self):
        current_hand = self.players[self.current_player].hand

        other_known_cards = []
        played_cards = []
        # This may be needed to change in the future if more players are added
        for player in self.players:
            if player != self.players[self.current_player]:
                other_known_cards.extend(player.known_cards) 
            played_cards.extend(player.won_cards)

        return {
            "hand": current_hand.cards,
            "trick": self.trick.played_cards,
            "current_player": self.current_player,
            "cards_left_in_deck": len(self.deck.cards),
            "other_players_known_cards": other_known_cards,
            "played_cards": played_cards
        }

