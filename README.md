This project utilizes MLP + attention layer ML model to play tressete card game.

It consists of;
    Game engine
    Unit tests
    CI tests
    Playing agents
    training pipeline
    ...tbc


Playing agents
Baseline player strategies for Tressette game engine, basically AI agents that pick cards based on simple handcrafted rules. These are starting points before training smarter AI.

Baseline Policies:
    Random Player: Plays a random valid card.

    Highest Lead Suit Player: If able to follow the lead suit, plays the highest card of that suit; else plays random.

    Simple Heuristic Player:
        If leading, play the highest card in hand.
        If following, try to beat the highest card played so far if possible, else play the lowest valid card.