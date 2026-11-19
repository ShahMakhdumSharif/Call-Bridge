"""Unit tests for Call Bridge playing and bidding rules."""

import pytest
from game.card import Card, Suit
from game.rules import (
    determine_trick_winner,
    get_legal_moves,
    is_legal_move,
    is_valid_bid,
    next_player,
    validate_bid,
)


def test_bid_validation() -> None:
    for b in range(0, 14):
        assert is_valid_bid(b) is True
        validate_bid(b)

    for invalid in (-1, 14, 99, 3.5, "4", None, True, False):
        assert is_valid_bid(invalid) is False  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            validate_bid(invalid)  # type: ignore[arg-type]


def test_follow_suit_rule_when_holding_lead_suit() -> None:
    hand = [
        Card.from_str("2H"),
        Card.from_str("KH"),
        Card.from_str("AS"),
        Card.from_str("9D"),
    ]
    legal = get_legal_moves(hand, Suit.HEARTS)
    assert set(legal) == {Card.from_str("2H"), Card.from_str("KH")}
    assert is_legal_move(Card.from_str("2H"), hand, Suit.HEARTS) is True
    assert is_legal_move(Card.from_str("AS"), hand, Suit.HEARTS) is False
    assert is_legal_move(Card.from_str("9D"), hand, Suit.HEARTS) is False


def test_void_in_lead_suit_allows_any_card() -> None:
    hand = [
        Card.from_str("2H"),
        Card.from_str("AS"),
        Card.from_str("9D"),
    ]
    legal = get_legal_moves(hand, Suit.CLUBS)
    assert set(legal) == set(hand)
    for c in hand:
        assert is_legal_move(c, hand, Suit.CLUBS) is True


def test_leading_trick_allows_any_card() -> None:
    hand = [
        Card.from_str("2H"),
        Card.from_str("AS"),
        Card.from_str("9D"),
    ]
    legal = get_legal_moves(hand, None)
    assert set(legal) == set(hand)


def test_trick_winner_no_spades_highest_lead_suit_wins() -> None:
    trick = [
        (0, Card.from_str("10H")),
        (1, Card.from_str("AD")),  # Off-suit non-trump cannot win
        (2, Card.from_str("KH")),  # Highest heart
        (3, Card.from_str("AC")),  # Off-suit non-trump cannot win
    ]
    assert determine_trick_winner(trick) == 2


def test_trick_winner_single_spade_trumps_lead_suit() -> None:
    trick = [
        (0, Card.from_str("AH")),
        (1, Card.from_str("KH")),
        (2, Card.from_str("2S")),  # Lowest spade beats Ace of Hearts
        (3, Card.from_str("QH")),
    ]
    assert determine_trick_winner(trick) == 2


def test_trick_winner_highest_spade_wins_among_multiple_spades() -> None:
    trick = [
        (0, Card.from_str("AD")),
        (1, Card.from_str("5S")),
        (2, Card.from_str("KS")),
        (3, Card.from_str("9S")),
    ]
    assert determine_trick_winner(trick) == 2


def test_clockwise_turn_order() -> None:
    assert next_player(0) == 1
    assert next_player(1) == 2
    assert next_player(2) == 3
    assert next_player(3) == 0
