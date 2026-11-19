"""Rollout policies for MCTS simulations in Call Bridge."""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Tuple
from game.card import Card, Suit, TRUMP_SUIT
from game.rules import determine_current_winning_play
from game.state import SimulationState


def card_beats(candidate: Card, current_best: Card, lead_suit: Suit) -> bool:
    """Return True if `candidate` beats `current_best` under `lead_suit` and Spades trump."""
    if candidate.suit == TRUMP_SUIT and current_best.suit != TRUMP_SUIT:
        return True
    if candidate.suit == current_best.suit:
        return candidate.value > current_best.value
    return False


def select_rollout_action(
    state: SimulationState,
    legal_moves: Sequence[Card],
    rng: random.Random,
    policy: str = "strategic",
    epsilon: float = 0.15,
) -> Card:
    """Choose a legal card during an MCTS rollout according to `policy`.

    Supported policies:
    - "strategic": Goal-directed play balancing winning needed tricks vs avoiding overtrick penalties.
    - "tactical_conservative": Prioritizes saving trumps and strictly avoiding overtricks once bid is met.
    - "random": Uniformly random legal card.
    """
    if len(legal_moves) == 1:
        return legal_moves[0]

    if policy == "random" or rng.random() < epsilon:
        return rng.choice(legal_moves)

    pid = state.current_player
    bid = state.bids.get(pid, 3)
    tricks_won = state.tricks_won.get(pid, 0)
    needed = bid - tricks_won
    current_trick = state.current_trick

    if needed > 0:
        # Player still needs tricks to fulfill their bid
        return _select_to_win_trick(
            legal_moves=legal_moves,
            current_trick=current_trick,
            conserve_trumps=(policy == "tactical_conservative"),
        )

    # Player has already achieved their bid (needed <= 0); avoid excess overtricks!
    return _select_to_duck_trick(
        legal_moves=legal_moves,
        current_trick=current_trick,
    )


def _select_to_win_trick(
    legal_moves: Sequence[Card],
    current_trick: Sequence[Tuple[int, Card]],
    conserve_trumps: bool = False,
) -> Card:
    """Select a card aiming to win the trick efficiently or discard lowest loser."""
    if not current_trick:
        # Leading the trick: lead highest non-trump winner (Ace/King) or lowest card
        non_trumps = [c for c in legal_moves if c.suit != TRUMP_SUIT]
        if non_trumps:
            high_non_trump = max(non_trumps, key=lambda c: c.value)
            if high_non_trump.value >= 13:  # K or A
                return high_non_trump
            return min(non_trumps, key=lambda c: c.value)
        return max(legal_moves, key=lambda c: c.value)

    lead_suit = current_trick[0][1].suit
    winning_play = determine_current_winning_play(current_trick)
    assert winning_play is not None
    _, current_best = winning_play

    winning_cards = [c for c in legal_moves if card_beats(c, current_best, lead_suit)]
    if winning_cards:
        if conserve_trumps and lead_suit != TRUMP_SUIT:
            non_trump_winners = [c for c in winning_cards if c.suit != TRUMP_SUIT]
            if non_trump_winners:
                return min(non_trump_winners, key=lambda c: c.value)
        # If last to play in trick (3 cards already on table), win as cheaply as possible
        if len(current_trick) == 3:
            return min(winning_cards, key=lambda c: (c.suit == TRUMP_SUIT, c.value))
        # Otherwise play a strong winning card
        return max(winning_cards, key=lambda c: c.value)

    # Cannot win the trick: slough lowest non-trump if possible, else lowest card
    non_trumps = [c for c in legal_moves if c.suit != TRUMP_SUIT]
    if non_trumps:
        return min(non_trumps, key=lambda c: c.value)
    return min(legal_moves, key=lambda c: c.value)


def _select_to_duck_trick(
    legal_moves: Sequence[Card],
    current_trick: Sequence[Tuple[int, Card]],
) -> Card:
    """Select a card aiming to avoid winning the trick (to prevent overtrick penalties)."""
    if not current_trick:
        # Forced to lead: lead the lowest non-trump card to minimize chance of winning
        non_trumps = [c for c in legal_moves if c.suit != TRUMP_SUIT]
        if non_trumps:
            return min(non_trumps, key=lambda c: c.value)
        return min(legal_moves, key=lambda c: c.value)

    lead_suit = current_trick[0][1].suit
    winning_play = determine_current_winning_play(current_trick)
    assert winning_play is not None
    _, current_best = winning_play

    losing_cards = [c for c in legal_moves if not card_beats(c, current_best, lead_suit)]
    if losing_cards:
        # Safe to dump our highest losing card so we don't get stuck winning later tricks!
        return max(losing_cards, key=lambda c: (c.suit == TRUMP_SUIT, c.value))

    # Forced to beat current best: play the lowest winning card hoping a subsequent player beats it
    return min(legal_moves, key=lambda c: c.value)


def run_rollout(
    state: SimulationState,
    rng: random.Random,
    policy: str = "strategic",
    cumulative_scores: Optional[Dict[int, int]] = None,
) -> Dict[int, float]:
    """Simulate from `state` to the end of the round and return normalized rewards per player."""
    while not state.is_terminal():
        legal = state.get_legal_moves()
        card = select_rollout_action(state, legal, rng=rng, policy=policy)
        state.play_card(card)

    round_scores = state.get_round_scores()
    rewards: Dict[int, float] = {}
    for pid, r_score in round_scores.items():
        # Normalize round score from [-13, +13] into roughly [0.0, 1.0] for UCT stability
        norm = (r_score + 13.0) / 26.0
        if cumulative_scores is not None:
            new_total = cumulative_scores.get(pid, 0) + r_score
            if new_total >= 16:
                norm += 0.35  # Bonus for reaching game-winning threshold
        rewards[pid] = norm

    return rewards
