"""Centralized scoring and winning-condition module for Call Bridge."""

from __future__ import annotations
from typing import Dict, List, Mapping, Tuple
from game.rules import TRICKS_PER_ROUND, validate_bid

WINNING_SCORE = 16


def calculate_player_score(bid: int, tricks_won: int) -> int:
    """Calculate the round score for a single player.

    Formula:
    - t < b      -> -b
    - t == b     -> +b
    - t == b + 1 -> 0
    - t > b + 1  -> -(t - b - 1)
    """
    validate_bid(bid)
    if isinstance(tricks_won, bool) or not isinstance(tricks_won, int) or not (0 <= tricks_won <= TRICKS_PER_ROUND):
        raise ValueError(f"Invalid tricks_won: {tricks_won!r}. Must be an integer between 0 and {TRICKS_PER_ROUND}.")

    if tricks_won < bid:
        return -bid
    if tricks_won == bid:
        return bid
    if tricks_won == bid + 1:
        return 0
    return -(tricks_won - bid - 1)


def calculate_round_scores(bids: Mapping[int, int], tricks_won: Mapping[int, int]) -> Dict[int, int]:
    """Calculate scores for all players after all 13 tricks of a round are completed."""
    if set(bids.keys()) != set(tricks_won.keys()):
        raise ValueError("Player IDs in bids and tricks_won must match.")
    total_tricks = sum(tricks_won.values())
    if total_tricks != TRICKS_PER_ROUND:
        raise ValueError(
            f"Scores can only be calculated after all {TRICKS_PER_ROUND} tricks are completed (got {total_tricks})."
        )

    return {pid: calculate_player_score(bids[pid], tricks_won[pid]) for pid in bids}


def update_cumulative_scores(
    cumulative_scores: Mapping[int, int],
    round_scores: Mapping[int, int],
) -> Dict[int, int]:
    """Return updated cumulative scores after adding round scores."""
    if set(cumulative_scores.keys()) != set(round_scores.keys()):
        raise ValueError("Player IDs in cumulative_scores and round_scores must match.")
    return {pid: cumulative_scores[pid] + round_scores[pid] for pid in cumulative_scores}


def check_winning_condition(
    cumulative_scores: Mapping[int, int],
    winning_score: int = WINNING_SCORE,
) -> Tuple[bool, List[int]]:
    """Check if the game has ended and identify the winner(s).

    Rules:
    - A player qualifies to win when their cumulative score reaches or exceeds `winning_score` (16).
    - If multiple players reach or exceed 16 in the same round:
      1. The player with the highest cumulative score wins.
      2. If the highest scores are tied, declare a tie (multiple winner IDs returned).

    Returns:
        (is_game_over, winner_ids)
    """
    qualifying = {pid: score for pid, score in cumulative_scores.items() if score >= winning_score}
    if not qualifying:
        return False, []

    highest_score = max(qualifying.values())
    winners = sorted(pid for pid, score in qualifying.items() if score == highest_score)
    return True, winners
