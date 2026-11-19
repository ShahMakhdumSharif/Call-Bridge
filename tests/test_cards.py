"""Unit tests for Card and Deck modules."""

import random
import pytest
from game.card import Card, Rank, Suit, TRUMP_SUIT
from game.deck import Deck


def test_standard_deck_has_52_unique_cards() -> None:
    deck = Deck()
    assert len(deck.cards) == 52
    assert len(set(deck.cards)) == 52


def test_trump_suit_is_permanently_spades() -> None:
    assert TRUMP_SUIT == Suit.SPADES
    spade_card = Card.from_str("2S")
    heart_card = Card.from_str("AH")
    assert spade_card.is_trump is True
    assert heart_card.is_trump is False


def test_rank_ordering_within_suit() -> None:
    expected_ranks = [
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
        Rank.ACE,
    ]
    cards = [Card(Suit.SPADES, r) for r in expected_ranks]
    for i in range(len(cards) - 1):
        assert cards[i].value < cards[i + 1].value
        assert cards[i] < cards[i + 1]


def test_deal_distributes_13_unique_cards_to_each_player() -> None:
    deck = Deck(rng=random.Random(42))
    deck.shuffle()
    hands = deck.deal()

    assert len(hands) == 4
    all_dealt = []
    for pid in range(4):
        assert len(hands[pid]) == 13
        all_dealt.extend(hands[pid])

    assert len(all_dealt) == 52
    assert len(set(all_dealt)) == 52


def test_card_from_str_parsing() -> None:
    c1 = Card.from_str("AS")
    assert c1 == Card(Suit.SPADES, Rank.ACE)
    c2 = Card.from_str("10♥")
    assert c2 == Card(Suit.HEARTS, Rank.TEN)
    c3 = Card.from_str("2C")
    assert c3 == Card(Suit.CLUBS, Rank.TWO)

    with pytest.raises(ValueError):
        Card.from_str("")
    with pytest.raises(ValueError):
        Card.from_str("15S")
    with pytest.raises(ValueError):
        Card.from_str("AX")
