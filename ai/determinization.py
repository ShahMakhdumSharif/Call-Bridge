"""Imperfect-information determinization for MCTS and Max^n agents.

Strictly operates ONLY on a player's restricted `PlayerObservation`.
Never accesses internal game engine state or actual hidden opponent hands.
"""

from __future__ import annotations
import random
from typing import Dict, List, Set
from game.card import Card, Suit
from game.deck import Deck
from game.rules import NUM_PLAYERS
from game.state import PlayerObservation, SimulationState


def get_unknown_cards(observation: PlayerObservation) -> List[Card]:
    """Return all cards that are neither in the observer's hand nor already played."""
    full_deck = set(Deck.create_standard_deck())
    known_cards = set(observation.hand) | set(observation.cards_played)
    unknown = list(full_deck - known_cards)
    unknown.sort()
    return unknown


def sample_determinization(
    observation: PlayerObservation,
    rng: random.Random,
) -> SimulationState:
    """Sample a plausible world state (SimulationState) consistent with `observation`.

    Guarantees:
    1. The observing player's hand is exactly `list(observation.hand)`.
    2. Every played card in `observation.cards_played` is excluded from all hands.
    3. Every opponent `pid` receives exactly `observation.remaining_card_counts[pid]` cards.
    4. No opponent receives a card of a suit in `observation.known_void_suits[pid]`.
    5. Uses only public information + the observer's private hand.
    """
    observer_id = observation.player_id
    unknown_cards = get_unknown_cards(observation)

    opponents = [pid for pid in range(NUM_PLAYERS) if pid != observer_id]
    needed_counts: Dict[int, int] = {
        pid: observation.remaining_card_counts[pid] for pid in opponents
    }

    total_needed = sum(needed_counts.values())
    if len(unknown_cards) != total_needed:
        raise ValueError(
            f"Determinization card count mismatch: {len(unknown_cards)} unknown cards "
            f"vs {total_needed} needed by opponents."
        )

    known_voids: Dict[int, Set[Suit]] = {
        pid: set(observation.known_void_suits.get(pid, frozenset()))
        for pid in opponents
    }

    assigned_hands = _assign_cards_respecting_voids(
        unknown_cards=unknown_cards,
        opponents=opponents,
        needed_counts=needed_counts,
        known_voids=known_voids,
        rng=rng,
    )
    assigned_hands[observer_id] = list(observation.hand)

    # Resolve public bids (defaulting any unsubmitted bid to a neutral estimate of 3)
    bids: Dict[int, int] = {}
    for pid in range(NUM_PLAYERS):
        b = observation.public_bids.get(pid)
        bids[pid] = int(b) if b is not None else 3

    return SimulationState(
        hands=assigned_hands,
        bids=bids,
        tricks_won=dict(observation.tricks_won),
        current_trick=list(observation.current_trick),
        current_player=observation.current_player,
        trick_number=observation.trick_number,
        trump_suit=observation.trump_suit,
    )


def _assign_cards_respecting_voids(
    unknown_cards: List[Card],
    opponents: List[int],
    needed_counts: Dict[int, int],
    known_voids: Dict[int, Set[Suit]],
    rng: random.Random,
) -> Dict[int, List[Card]]:
    """Assign `unknown_cards` to `opponents` respecting void-suit constraints."""
    has_constraints = any(len(known_voids[pid]) > 0 for pid in opponents)

    if not has_constraints:
        shuffled = list(unknown_cards)
        rng.shuffle(shuffled)
        hands: Dict[int, List[Card]] = {}
        offset = 0
        for pid in opponents:
            count = needed_counts[pid]
            hands[pid] = shuffled[offset : offset + count]
            offset += count
        return hands

    # Try constrained card-by-card assignment (most constrained cards first)
    for _ in range(12):
        shuffled = list(unknown_cards)
        rng.shuffle(shuffled)
        # Sort cards by fewest eligible opponents (most constrained first)
        shuffled.sort(
            key=lambda c: sum(1 for pid in opponents if c.suit not in known_voids[pid])
        )

        hands = {pid: [] for pid in opponents}
        remaining_slots = dict(needed_counts)
        success = True

        for card in shuffled:
            eligible = [
                pid
                for pid in opponents
                if remaining_slots[pid] > 0 and card.suit not in known_voids[pid]
            ]
            if not eligible:
                success = False
                break
            # Weight selection by number of remaining slots for uniform distribution
            weights = [remaining_slots[pid] for pid in eligible]
            chosen_pid = rng.choices(eligible, weights=weights, k=1)[0]
            hands[chosen_pid].append(card)
            remaining_slots[chosen_pid] -= 1

        if success:
            return hands

    # Backtracking solver with Hall's capacity pruning for guaranteed valid assignment
    shuffled = list(unknown_cards)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda c: sum(1 for pid in opponents if c.suit not in known_voids[pid])
    )
    hands = {pid: [] for pid in opponents}
    remaining_slots = dict(needed_counts)

    # Precompute suffix suit counts for fast capacity feasibility check
    n_cards = len(shuffled)

    def can_complete(idx: int) -> bool:
        for pid in opponents:
            if remaining_slots[pid] == 0:
                continue
            # Count how many remaining cards from idx..n_cards can legally go to pid
            compatible = sum(
                1 for k in range(idx, n_cards) if shuffled[k].suit not in known_voids[pid]
            )
            if compatible < remaining_slots[pid]:
                return False
        return True

    def backtrack(idx: int) -> bool:
        if idx == n_cards:
            return True
        if not can_complete(idx):
            return False
        card = shuffled[idx]
        eligible = [
            pid
            for pid in opponents
            if remaining_slots[pid] > 0 and card.suit not in known_voids[pid]
        ]
        rng.shuffle(eligible)
        for pid in eligible:
            hands[pid].append(card)
            remaining_slots[pid] -= 1
            if backtrack(idx + 1):
                return True
            remaining_slots[pid] += 1
            hands[pid].pop()
        return False

    if backtrack(0):
        return hands

    # Fallback if constraints are unsatisfiable (never occurs in valid game history)
    shuffled = list(unknown_cards)
    rng.shuffle(shuffled)
    hands = {}
    offset = 0
    for pid in opponents:
        count = needed_counts[pid]
        hands[pid] = shuffled[offset : offset + count]
        offset += count
    return hands
