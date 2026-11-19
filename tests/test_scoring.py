"""Unit tests for Call Bridge scoring and winning condition rules."""

import pytest
from game.scoring import (
    WINNING_SCORE,
    calculate_player_score,
    calculate_round_scores,
    check_winning_condition,
    update_cumulative_scores,
)


def test_example_bid_4_table_from_specification() -> None:
    """Verify the exact table from Section 4 of the specification for Bid = 4."""
    expected = {
        2: -4,
        3: -4,
        4: 4,
        5: 0,
        6: -1,
        7: -2,
        8: -3,
    }
    for tricks_won, expected_score in expected.items():
        assert calculate_player_score(4, tricks_won) == expected_score


def test_bid_zero_edge_cases() -> None:
    assert calculate_player_score(0, 0) == 0
    assert calculate_player_score(0, 1) == 0
    assert calculate_player_score(0, 2) == -1
    assert calculate_player_score(0, 5) == -4


def test_bid_thirteen_edge_cases() -> None:
    for tricks in range(0, 13):
        assert calculate_player_score(13, tricks) == -13
    assert calculate_player_score(13, 13) == 13


def test_round_scores_require_all_13_tricks() -> None:
    bids = {0: 3, 1: 4, 2: 2, 3: 4}
    incomplete_tricks = {0: 3, 1: 4, 2: 2, 3: 3}  # Sum = 12
    with pytest.raises(ValueError):
        calculate_round_scores(bids, incomplete_tricks)

    complete_tricks = {0: 3, 1: 5, 2: 1, 3: 4}  # Sum = 13
    scores = calculate_round_scores(bids, complete_tricks)
    assert scores == {
        0: 3,   # t == b
        1: 0,   # t == b + 1
        2: -2,  # t < b
        3: 4,   # t == b
    }


def test_cumulative_scores_update() -> None:
    cumulative = {0: 10, 1: 8, 2: -2, 3: 12}
    round_scores = {0: 4, 1: -3, 2: 5, 3: 4}
    updated = update_cumulative_scores(cumulative, round_scores)
    assert updated == {0: 14, 1: 5, 2: 3, 3: 16}


def test_winning_condition_exact_16_and_exceeding_16() -> None:
    assert WINNING_SCORE == 16

    is_over, winners = check_winning_condition({0: 15, 1: 12, 2: 8, 3: 14})
    assert is_over is False
    assert winners == []

    # Exact 16
    is_over, winners = check_winning_condition({0: 16, 1: 12, 2: 8, 3: 14})
    assert is_over is True
    assert winners == [0]

    # Exceeding 16
    is_over, winners = check_winning_condition({0: 14, 1: 19, 2: 8, 3: 15})
    assert is_over is True
    assert winners == [1]


def test_winning_condition_multiple_players_over_16_highest_wins() -> None:
    is_over, winners = check_winning_condition({0: 17, 1: 19, 2: 16, 3: 10})
    assert is_over is True
    assert winners == [1]


def test_winning_condition_tie_at_highest_score() -> None:
    is_over, winners = check_winning_condition({0: 18, 1: 16, 2: 18, 3: 12})
    assert is_over is True
    assert winners == [0, 2]
