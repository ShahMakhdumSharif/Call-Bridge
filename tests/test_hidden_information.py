"""Automated tests for AI Information Security and Imperfect Information (Section 7)."""

import random
import pytest
from ai.determinization import get_unknown_cards, sample_determinization
from game.card import Card, Suit
from game.engine import GameEngine
from players.mcts_agent import MCTSAgent


def test_observation_does_not_contain_opponents_private_cards() -> None:
    """Verify PlayerObservation never contains or references opponents' cards."""
    engine = GameEngine(seed=101)

    for pid in range(4):
        obs = engine.get_observation(pid)
        own_cards = set(obs.hand)
        assert len(own_cards) == 13

        # Collect all cards from the other 3 players
        opponent_cards = set()
        for other_id in range(4):
            if other_id != pid:
                opponent_cards.update(engine.get_observation(other_id).hand)

        assert len(opponent_cards) == 39
        assert own_cards.isdisjoint(opponent_cards)

        # Inspect every field of obs to verify no opponent card is hidden inside any attribute
        for field_name, field_value in vars(obs).items():
            if isinstance(field_value, (tuple, list, set, frozenset)):
                for item in field_value:
                    if isinstance(item, Card):
                        assert item not in opponent_cards, (
                            f"Opponent card {item} leaked in observation.{field_name}"
                        )


def test_engine_does_not_expose_internal_state_publicly() -> None:
    """Verify AI agents cannot retrieve internal hands via public engine attributes."""
    engine = GameEngine(seed=202)
    public_attrs = [attr for attr in dir(engine) if not attr.startswith("_")]
    assert "hands" not in public_attrs
    assert "state" not in public_attrs

    results = engine.get_game_results()
    assert "hands" not in results


def test_determinization_uses_only_legitimate_information_and_respects_voids() -> None:
    """Verify determinization reconstructs valid hands without knowing actual hidden hands."""
    engine = GameEngine(seed=303, initial_dealer=3)
    # Create a controlled scenario where Player 1 fails to follow Hearts (known void in Hearts)
    custom_hands = {
        0: [Card.from_str(c) for c in ["2H", "3H", "4H", "5H", "2S", "3S", "4S", "2D", "3D", "4D", "2C", "3C", "4C"]],
        1: [Card.from_str(c) for c in ["5S", "6S", "7S", "8S", "9S", "5D", "6D", "7D", "8D", "5C", "6C", "7C", "8C"]],  # 0 Hearts!
        2: [Card.from_str(c) for c in ["6H", "7H", "8H", "9H", "10S", "JS", "QS", "9D", "10D", "JD", "9C", "10C", "JC"]],
        3: [Card.from_str(c) for c in ["10H", "JH", "QH", "KH", "AH", "KS", "AS", "QD", "KD", "AD", "QC", "KC", "AC"]],
    }
    engine._GameEngine__state.round_number = 0
    engine.start_new_round(custom_hands=custom_hands, starting_player=0)

    for pid in range(4):
        engine.submit_bid(pid, 3)

    # Trick 1: Player 0 leads 2H; Player 1 has no Hearts and plays 5C (revealing void in Hearts!)
    engine.play_card(0, Card.from_str("2H"))
    engine.play_card(1, Card.from_str("5C"))
    engine.play_card(2, Card.from_str("6H"))
    engine.play_card(3, Card.from_str("10H"))

    obs0 = engine.get_observation(0)
    assert Suit.HEARTS in obs0.known_void_suits[1]

    rng = random.Random(42)
    for _ in range(30):
        sim = sample_determinization(obs0, rng)
        # Player 0 has exact own hand
        assert set(sim.hands[0]) == set(obs0.hand)
        # Played cards are never in any hand
        for pid in range(4):
            assert set(sim.hands[pid]).isdisjoint(set(obs0.cards_played))
            assert len(sim.hands[pid]) == obs0.remaining_card_counts[pid]
        # Player 1 is known void in Hearts, so Player 1 must have 0 Hearts in every determinization
        assert all(c.suit != Suit.HEARTS for c in sim.hands[1])


def test_swapping_opponents_hidden_cards_does_not_change_observation_or_leak_cards() -> None:
    """Verify changing opponents' hidden cards without changing Player 0's hand produces
    identical observations and identical deterministic AI behavior for a fixed seed.
    """
    hands_world_a = {
        0: [Card.from_str(c) for c in ["AS", "KS", "AH", "KH", "AD", "KD", "AC", "KC", "2S", "3S", "4H", "5D", "6C"]],
        1: [Card.from_str(c) for c in ["QS", "JS", "10S", "9S", "8S", "7S", "6S", "5S", "4S", "2H", "3H", "5H", "6H"]],
        2: [Card.from_str(c) for c in ["QH", "JH", "10H", "9H", "8H", "7H", "QD", "JD", "10D", "9D", "8D", "7D", "6D"]],
        3: [Card.from_str(c) for c in ["4D", "3D", "2D", "QC", "JC", "10C", "9C", "8C", "7C", "5C", "4C", "3C", "2C"]],
    }
    # World B swaps all cards between Player 1 and Player 2 while keeping Player 0's hand identical
    hands_world_b = {
        0: list(hands_world_a[0]),
        1: list(hands_world_a[2]),
        2: list(hands_world_a[1]),
        3: list(hands_world_a[3]),
    }

    engine_a = GameEngine(seed=1, initial_dealer=3)
    engine_a._GameEngine__state.round_number = 0
    engine_a.start_new_round(custom_hands=hands_world_a, starting_player=0)

    engine_b = GameEngine(seed=1, initial_dealer=3)
    engine_b._GameEngine__state.round_number = 0
    engine_b.start_new_round(custom_hands=hands_world_b, starting_player=0)

    for pid in range(4):
        engine_a.submit_bid(pid, 3)
        engine_b.submit_bid(pid, 3)

    obs_a = engine_a.get_observation(0)
    obs_b = engine_b.get_observation(0)

    # Observations must be 100% identical
    assert obs_a == obs_b
    assert get_unknown_cards(obs_a) == get_unknown_cards(obs_b)

    # An MCTS agent with the same seed must make the exact same decision on both
    agent_a = MCTSAgent.create_agent_1(player_id=0, simulation_budget=100, seed=555)
    agent_b = MCTSAgent.create_agent_1(player_id=0, simulation_budget=100, seed=555)

    assert agent_a.choose_card(obs_a) == agent_b.choose_card(obs_b)
