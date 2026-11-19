"""Centralized Call Bridge playing and bidding rules."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from game.card import Card, Suit, TRUMP_SUIT

NUM_PLAYERS = 4
TRICKS_PER_ROUND = 13
MIN_BID = 0
MAX_BID = 13


def is_valid_bid(bid: int) -> bool:
    """Return True if bid is an integer between 0 and 13 inclusive."""
    if isinstance(bid, bool) or not isinstance(bid, int):
        return False
    return MIN_BID <= bid <= MAX_BID


def validate_bid(bid: int) -> None:
    """Validate that a bid is an integer in [0, 13], raising ValueError otherwise."""
    if not is_valid_bid(bid):
        raise ValueError(f"Invalid bid: {bid!r}. Bid must be an integer between {MIN_BID} and {MAX_BID}.")


def next_player(current_player: int, num_players: int = NUM_PLAYERS) -> int:
    """Return the next player index in clockwise turn order."""
    if not (0 <= current_player < num_players):
        raise ValueError(f"Invalid player index: {current_player}")
    return (current_player + 1) % num_players


def get_legal_moves(hand: Sequence[Card], lead_suit: Optional[Suit]) -> List[Card]:
    """Return all legal cards a player can play given their hand and the current trick's lead suit.

    Rules:
    - If lead_suit is None (player is leading the trick), any card in hand may be played.
    - If lead_suit is set, the player must play a card of lead_suit if they hold at least one.
    - If the player has no cards of lead_suit, they may play any card from their hand.
    """
    if not hand:
        return []
    if lead_suit is None:
        return list(hand)

    same_suit_cards = [card for card in hand if card.suit == lead_suit]
    if same_suit_cards:
        return same_suit_cards
    return list(hand)


def is_legal_move(card: Card, hand: Sequence[Card], lead_suit: Optional[Suit]) -> bool:
    """Return True if playing `card` from `hand` is legal under `lead_suit`."""
    if card not in hand:
        return False
    return card in get_legal_moves(hand, lead_suit)


def determine_trick_winner(trick_cards: Sequence[Tuple[int, Card]]) -> int:
    """Determine the winning player ID for a completed 4-card trick.

    Rules:
    - If one or more Spades (trump) are played, the highest Spade wins.
    - Otherwise, the highest card of the lead suit wins.
    """
    if len(trick_cards) != NUM_PLAYERS:
        raise ValueError(f"Trick must have exactly {NUM_PLAYERS} cards, got {len(trick_cards)}.")

    lead_suit = trick_cards[0][1].suit

    trump_plays = [(pid, card) for pid, card in trick_cards if card.suit == TRUMP_SUIT]
    if trump_plays:
        winner_pid, _ = max(trump_plays, key=lambda item: item[1].value)
        return winner_pid

    lead_suit_plays = [(pid, card) for pid, card in trick_cards if card.suit == lead_suit]
    winner_pid, _ = max(lead_suit_plays, key=lambda item: item[1].value)
    return winner_pid


def determine_current_winning_play(trick_cards: Sequence[Tuple[int, Card]]) -> Optional[Tuple[int, Card]]:
    """Determine who is currently winning a partially or fully played trick."""
    if not trick_cards:
        return None
    lead_suit = trick_cards[0][1].suit
    trump_plays = [(pid, card) for pid, card in trick_cards if card.suit == TRUMP_SUIT]
    if trump_plays:
        return max(trump_plays, key=lambda item: item[1].value)
    lead_suit_plays = [(pid, card) for pid, card in trick_cards if card.suit == lead_suit]
    return max(lead_suit_plays, key=lambda item: item[1].value)
