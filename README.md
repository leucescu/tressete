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

State input:
    | Feature                  | Description                                     | Shape         |
    | ------------------------ | ----------------------------------------------- | ------------- |
    | **Own hand**             | 40-dim binary — 1 where card is in hand         | 40            |
    | **Known cards (others)** | 40-dim binary — known cards not in agent's hand | 40            |
    | **Played cards**         | 40-dim binary — cards already played            | 40            |
    | **Current trick**        | 40-dim binary — cards currently on the table    | 40            |
    | **Points per player**    | Current score for each player                   | 2 floats      |
    | **Final trick flag**     | Is this the last trick? (0 or 1)                | 1 float       |
