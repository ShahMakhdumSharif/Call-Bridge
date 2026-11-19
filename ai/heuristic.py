"""Heuristic evaluation function for 4-player Max^n search in Call Bridge."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from game.card import Card, Rank, Suit, TRUMP_SUIT
from game.rules import NUM_PLAYERS, TRICKS_PER_ROUND, determine_current_winning_play, next_player
from game.scoring import WINNING_SCORE, calculate_player_score
from game.state import SimulationState


def evaluate_hand_potential(hand: List[Card]) -> float:
    """Compute a fast heuristic trick-winning weight for a remaining hand."""
    if not hand:
        return 0.0

    weight = 0.0
    suit_counts: Dict[Suit, int] = {s: 0 for s in Suit}
    for card in hand:
        suit_counts[card.suit] += 1
        if card.suit == TRUMP_SUIT:
            # Trump cards carry higher trick-winning power
            if card.rank == Rank.ACE:
                weight += 1.45
            elif card.rank == Rank.KING:
                weight += 1.20
            elif card.rank == Rank.QUEEN:
                weight += 0.95
            elif card.rank == Rank.JACK:
                weight += 0.75
            else:
                weight += 0.45 + (card.value - 2) * 0.03
        else:
            if card.rank == Rank.ACE:
                weight += 1.05
            elif card.rank == Rank.KING:
                weight += 0.68
            elif card.rank == Rank.QUEEN:
                weight += 0.38
            elif card.rank == Rank.JACK:
                weight += 0.18
            else:
                weight += (card.value - 2) * 0.015

    # Void side-suit ruffing bonus when holding trumps
    trumps = suit_counts[TRUMP_SUIT]
    if trumps > 0:
        for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS):
            if suit_counts[suit] == 0:
                weight += 0.25 * min(trumps, 2)

    return max(0.05, weight)


def _estimate_current_trick_probabilities(state: SimulationState) -> Dict[int, float]:
    """Estimate each player's probability of winning the currently open trick."""
    probs = {pid: 0.0 for pid in range(NUM_PLAYERS)}
    if not state.current_trick:
        return probs

    winning_play = determine_current_winning_play(state.current_trick)
    assert winning_play is not None
    current_winner, best_card = winning_play
    lead_suit = state.current_trick[0][1].suit

    # Identify players who have not yet played in this trick
    played_pids = {pid for pid, _ in state.current_trick}
    remaining_pids: List[int] = []
    curr = state.current_player
    while len(played_pids) + len(remaining_pids) < NUM_PLAYERS:
        remaining_pids.append(curr)
        curr = next_player(curr, NUM_PLAYERS)

    if not remaining_pids:
        probs[current_winner] = 1.0
        return probs

    # Check if remaining players in this trick hold cards that can beat `best_card`
    challenger_weights: Dict[int, float] = {}
    for pid in remaining_pids:
        legal = state.get_legal_moves(pid)
        beating = [
            c
            for c in legal
            if (c.suit == TRUMP_SUIT and best_card.suit != TRUMP_SUIT)
            or (c.suit == best_card.suit and c.value > best_card.value)
        ]
        if beating:
            highest_beater = max(beating, key=lambda c: (c.suit == TRUMP_SUIT, c.value))
            strength = 1.0 + (0.4 if highest_beater.suit == TRUMP_SUIT else 0.0) + highest_beater.value / 14.0
            # Check if this player actually wants to win the trick
            needed = state.bids.get(pid, 3) - state.tricks_won.get(pid, 0)
            desire = 1.25 if needed > 0 else 0.35
            challenger_weights[pid] = strength * desire

    if not challenger_weights:
        probs[current_winner] = 1.0
        return probs

    leader_weight = 1.1 + (0.5 if best_card.suit == TRUMP_SUIT else 0.0) + (best_card.value / 14.0)
    total_w = leader_weight + sum(challenger_weights.values())
    probs[current_winner] = leader_weight / total_w
    for pid, w in challenger_weights.items():
        probs[pid] = w / total_w
    return probs


def expected_score_from_continuous_tricks(bid: int, expected_tricks: float) -> float:
    """Smoothly evaluate the Call Bridge scoring function for continuous expected tricks."""
    clamped_t = max(0.0, min(float(TRICKS_PER_ROUND), expected_tricks))
    low_t = int(math.floor(clamped_t))
    high_t = int(math.ceil(clamped_t))
    frac = clamped_t - low_t

    score_low = float(calculate_player_score(bid, low_t))
    score_high = float(calculate_player_score(bid, high_t))
    blended_score = (1.0 - frac) * score_low + frac * score_high

    # Add smooth shaping so the heuristic prefers states moving closer to `bid` (or `bid + 0.3`)
    distance_to_target = abs(clamped_t - (bid + 0.15))
    shaping = -0.35 * distance_to_target
    return blended_score + shaping


def evaluate_state(
    state: SimulationState,
    cumulative_scores: Optional[Dict[int, int]] = None,
) -> Tuple[float, float, float, float]:
    """Evaluate `state` and return a 4-tuple of utilities `(u_0, u_1, u_2, u_3)`.

    Considers:
    - Bids and tricks already won.
    - Current trick cards and who can beat the current winning card.
    - Remaining cards and trump cards (Spades).
    - Exact centralized scoring formula and proximity to the 16-point winning threshold.
    """
    if state.is_terminal():
        round_scores = state.get_round_scores()
        utils: List[float] = []
        for pid in range(NUM_PLAYERS):
            u = float(round_scores[pid])
            if cumulative_scores is not None:
                total = cumulative_scores.get(pid, 0) + round_scores[pid]
                if total >= WINNING_SCORE:
                    u += 10.0
            utils.append(u)
        return (utils[0], utils[1], utils[2], utils[3])

    # 1. Tricks already won
    expected_tricks: Dict[int, float] = {
        pid: float(state.tricks_won.get(pid, 0)) for pid in range(NUM_PLAYERS)
    }

    tricks_completed = sum(state.tricks_won.values())
    open_trick_count = 1 if state.current_trick else 0
    future_full_tricks = max(0, TRICKS_PER_ROUND - tricks_completed - open_trick_count)

    # 2. Current open trick share
    if open_trick_count == 1:
        trick_probs = _estimate_current_trick_probabilities(state)
        for pid in range(NUM_PLAYERS):
            expected_tricks[pid] += trick_probs[pid]

    # 3. Future full tricks distributed proportionally to remaining hand potentials
    if future_full_tricks > 0:
        potentials = {pid: evaluate_hand_potential(state.hands[pid]) for pid in range(NUM_PLAYERS)}
        total_potential = sum(potentials.values())
        if total_potential > 0:
            for pid in range(NUM_PLAYERS):
                expected_tricks[pid] += future_full_tricks * (potentials[pid] / total_potential)
        else:
            for pid in range(NUM_PLAYERS):
                expected_tricks[pid] += future_full_tricks / float(NUM_PLAYERS)

    # 4. Convert each player's expected tricks into expected score utility
    raw_utilities: Dict[int, float] = {}
    for pid in range(NUM_PLAYERS):
        bid = state.bids.get(pid, 3)
        u = expected_score_from_continuous_tricks(bid, expected_tricks[pid])
        if cumulative_scores is not None:
            projected_total = cumulative_scores.get(pid, 0) + u
            if projected_total >= WINNING_SCORE:
                u += 6.0
        raw_utilities[pid] = u

    # Include slight relative margin over opponents so players break ties by hindering leading opponents
    avg_total = sum(raw_utilities.values()) / float(NUM_PLAYERS)
    final_utils = tuple(
        raw_utilities[pid] + 0.10 * (raw_utilities[pid] - avg_total)
        for pid in range(NUM_PLAYERS)
    )
    return (final_utils[0], final_utils[1], final_utils[2], final_utils[3])
