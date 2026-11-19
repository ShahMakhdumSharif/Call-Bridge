"""Intelligent bidding strategies for Call Bridge AI agents."""

from __future__ import annotations
from typing import Dict, List, Sequence
from game.card import Card, Rank, Suit, TRUMP_SUIT
from game.rules import MAX_BID, MIN_BID
from game.state import PlayerObservation


def estimate_hand_tricks(hand: Sequence[Card]) -> float:
    """Estimate expected number of tricks a 13-card hand can win in Call Bridge.

    Considers:
    - High-ranking cards (A, K, Q, J) in non-trump suits and their protection/length.
    - Spade (trump) count and high trump honors.
    - Suit distribution, long trump suits, and void/singleton ruffing potential.
    """
    suits_map: Dict[Suit, List[Card]] = {s: [] for s in Suit}
    for card in hand:
        suits_map[card.suit].append(card)
    for s in suits_map:
        suits_map[s].sort(key=lambda c: c.value, reverse=True)

    spades = suits_map[TRUMP_SUIT]
    num_spades = len(spades)
    spade_ranks = {c.rank for c in spades}

    expected_tricks = 0.0

    # 1. Evaluate non-trump suits (Clubs, Diamonds, Hearts)
    for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS):
        cards = suits_map[suit]
        length = len(cards)
        ranks = {c.rank for c in cards}

        if Rank.ACE in ranks:
            # Ace almost always wins unless trumped on round 1 (rare)
            expected_tricks += 0.92 if length <= 5 else 0.80
        if Rank.KING in ranks:
            if length >= 2 or Rank.ACE in ranks:
                # Guarded King
                expected_tricks += 0.62 if length <= 4 else 0.45
            else:
                # Singleton King
                expected_tricks += 0.20
        if Rank.QUEEN in ranks:
            if length >= 3 or (Rank.ACE in ranks and Rank.KING in ranks):
                expected_tricks += 0.32 if length <= 4 else 0.20
            elif length == 2 and (Rank.ACE in ranks or Rank.KING in ranks):
                expected_tricks += 0.25

    # 2. Evaluate Trump suit (Spades)
    if Rank.ACE in spade_ranks:
        expected_tricks += 0.98
    if Rank.KING in spade_ranks:
        expected_tricks += 0.85 if (num_spades >= 2 or Rank.ACE in spade_ranks) else 0.45
    if Rank.QUEEN in spade_ranks:
        if num_spades >= 3 or (Rank.ACE in spade_ranks and Rank.KING in spade_ranks):
            expected_tricks += 0.70
        elif num_spades == 2:
            expected_tricks += 0.40
    if Rank.JACK in spade_ranks and num_spades >= 4:
        expected_tricks += 0.45
    if Rank.TEN in spade_ranks and num_spades >= 5:
        expected_tricks += 0.35

    # 3. Extra trump length bonus (spades beyond the first 3 often win tricks once trumps are drawn)
    if num_spades > 3:
        extra_trumps = num_spades - 3
        # Avoid double-counting high honors already counted above
        low_spades = sum(1 for c in spades if c.rank.value <= Rank.NINE.value)
        length_bonus = min(extra_trumps, low_spades) * 0.68
        expected_tricks += length_bonus

    # 4. Ruffing potential from void or singleton side suits (only if holding enough trumps)
    spare_trumps = max(0, num_spades - 2)
    if spare_trumps > 0:
        ruff_bonus = 0.0
        for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS):
            length = len(suits_map[suit])
            if length == 0:
                ruff_bonus += 0.55
            elif length == 1:
                ruff_bonus += 0.28
        expected_tricks += min(ruff_bonus, spare_trumps * 0.60)

    return expected_tricks


def choose_strategic_bid(
    observation: PlayerObservation,
    risk_offset: float = -0.20,
) -> int:
    """Calculate an integer bid in [0, 13] from the player's observation.

    Because Call Bridge penalizes under-tricking (`t < b -> -b`) and allows
    `t == b + 1` with `0` penalty (while `t == b` scores `+b`), a slight negative
    `risk_offset` calibrates the bid so the agent achieves `+b` reliably.
    """
    raw_estimate = estimate_hand_tricks(observation.hand)

    # Consider cumulative score context: if close to 16, check if a safe achievable bid suffices
    my_score = observation.cumulative_scores.get(observation.player_id, 0)
    points_to_win = 16 - my_score

    adjusted = raw_estimate + risk_offset
    bid = int(round(adjusted))

    # If the agent needs 1..4 points to reach 16 and the hand strongly supports it, avoid bidding 0
    if 1 <= points_to_win <= 4 and raw_estimate >= points_to_win - 0.15:
        bid = max(bid, points_to_win)

    # Clamp to [MIN_BID, MAX_BID]
    return max(MIN_BID, min(MAX_BID, bid))
