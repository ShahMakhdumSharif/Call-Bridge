"""Unit tests for the central GameEngine."""

import os
import pytest
from ai.evaluation import run_match_experiment
from game.card import Card, Rank, Suit
from game.engine import GameEngine
from game.scoring import calculate_player_score
from game.state import GamePhase
from players.human_player import HumanPlayer
from players.maxn_agent import MaxNAgent
from players.mcts_agent import MCTSAgent
from players.random_agent import RandomAgent


def make_deterministic_hands() -> dict[int, list[Card]]:
    """Create 4 deterministic 13-card hands where each player holds one full suit."""
    suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
    return {
        pid: [Card(suits[pid], rank) for rank in Rank]
        for pid in range(4)
    }


def test_engine_initial_deal_and_bidding_sum_unrestricted() -> None:
    engine = GameEngine(seed=123, initial_dealer=3)
    assert engine.phase == GamePhase.BIDDING
    assert engine.round_number == 1
    assert engine.dealer == 3
    assert engine.starting_player == 0
    assert engine.current_player == 0

    for pid in range(4):
        obs = engine.get_observation(pid)
        assert len(obs.hand) == 13

    # Bids do not need to sum to 13 (e.g., all players bid 0 or all bid 13)
    for pid in range(4):
        engine.submit_bid(pid, 0)

    assert engine.phase == GamePhase.PLAYING
    assert engine.bids == {0: 0, 1: 0, 2: 0, 3: 0}


def test_engine_full_round_and_trick_winner_leads_next() -> None:
    engine = GameEngine(seed=10, initial_dealer=3)
    hands = make_deterministic_hands()
    # Reset round 1 with custom hands: Player 0 has all Spades, 1 Hearts, 2 Diamonds, 3 Clubs
    engine.start_new_game(initial_dealer=3)
    engine._GameEngine__state.round_number = 0  # reset counter to start round 1 with custom hands
    engine.start_new_round(custom_hands=hands, starting_player=1)

    # Bidding order starts from starting_player=1 -> 2 -> 3 -> 0
    engine.submit_bid(1, 0)
    engine.submit_bid(2, 0)
    engine.submit_bid(3, 0)
    engine.submit_bid(0, 13)

    assert engine.phase == GamePhase.PLAYING
    assert engine.current_player == 1

    # Play all 13 tricks; Player 0 has all Spades so Player 0 wins every trick
    for trick_idx in range(13):
        lead_pid = engine.current_player
        for step in range(4):
            pid = engine.current_player
            legal = engine.get_legal_moves(pid)
            assert len(legal) >= 1
            winner = engine.play_card(pid, legal[0])
            if step < 3:
                assert winner is None
            else:
                assert winner == 0
        if trick_idx < 12:
            # Winner of trick leads next trick
            assert engine.current_player == 0
            assert engine.phase == GamePhase.PLAYING

    # After 13 tricks: Player 0 won 13 tricks (bid 13 -> +13), others won 0 (bid 0 -> 0)
    assert engine.phase == GamePhase.ROUND_OVER
    assert engine.tricks_won == {0: 13, 1: 0, 2: 0, 3: 0}
    assert engine.round_scores == {0: 13, 1: 0, 2: 0, 3: 0}
    assert engine.cumulative_scores == {0: 13, 1: 0, 2: 0, 3: 0}
    assert engine.is_game_over is False


def test_engine_game_terminates_at_16_points_and_blocks_new_round() -> None:
    engine = GameEngine(seed=99, initial_dealer=3)
    hands = make_deterministic_hands()

    # Round 1: Player 0 bids 13 and wins 13 -> score +13
    engine._GameEngine__state.round_number = 0
    engine.start_new_round(custom_hands=hands, starting_player=0)
    for pid in range(4):
        engine.submit_bid(pid, 13 if pid == 0 else 0)
    for _ in range(13):
        for _ in range(4):
            pid = engine.current_player
            engine.play_card(pid, engine.get_legal_moves(pid)[0])

    assert engine.cumulative_scores[0] == 13
    assert engine.phase == GamePhase.ROUND_OVER

    # Round 2: Player 0 bids 13 and wins 13 -> cumulative score 26 >= 16 -> GAME_OVER
    engine.start_new_round(custom_hands=hands, starting_player=0)
    for pid in range(4):
        engine.submit_bid(pid, 13 if pid == 0 else 0)
    for _ in range(13):
        for _ in range(4):
            pid = engine.current_player
            engine.play_card(pid, engine.get_legal_moves(pid)[0])

    assert engine.cumulative_scores[0] == 26
    assert engine.phase == GamePhase.GAME_OVER
    assert engine.is_game_over is True
    assert engine.winner_ids == [0]

    # Must not allow starting another round after game over
    with pytest.raises(RuntimeError):
        engine.start_new_round()


def test_random_agent_complete_rounds_and_game_simulation() -> None:
    """Phase 2 verification: simulate rounds and complete game with RandomAgents."""
    engine = GameEngine(seed=42)
    agents = {pid: RandomAgent(player_id=pid, name=f"Random-{pid}", seed=100 + pid) for pid in range(4)}

    for round_idx in range(5):
        if round_idx > 0:
            engine.start_new_round()

        # Verify every player receives 13 unique cards
        seen_cards = set()
        for pid in range(4):
            obs = engine.get_observation(pid)
            assert len(obs.hand) == 13
            seen_cards.update(obs.hand)
        assert len(seen_cards) == 52

        # Bidding phase
        for _ in range(4):
            pid = engine.current_player
            obs = engine.get_observation(pid)
            bid = agents[pid].choose_bid(obs)
            assert 0 <= bid <= 13
            engine.submit_bid(pid, bid)

        assert engine.phase == GamePhase.PLAYING

        # Playing phase: 13 tricks
        for trick_num in range(1, 14):
            assert engine.trick_number == trick_num
            for _ in range(4):
                pid = engine.current_player
                obs = engine.get_observation(pid)
                card = agents[pid].choose_card(obs)
                assert card in obs.legal_moves
                engine.play_card(pid, card)

        # Verify all 13 tricks completed and scoring formula matches
        assert sum(engine.tricks_won.values()) == 13
        for pid in range(4):
            b = engine.bids[pid]
            t = engine.tricks_won[pid]
            assert b is not None
            assert engine.round_scores[pid] == calculate_player_score(b, t)
            assert len(engine.get_observation(pid).hand) == 0

        if engine.is_game_over:
            break


def test_complete_game_one_human_and_three_intelligent_ai_agents() -> None:
    """Phase 5 Integration Test: Simulate a complete multi-round game with
    Player 1 (Human), Player 2 (MCTS-1), Player 3 (MCTS-2), and Player 4 (Max^n)
    until a player reaches or exceeds 16 points.
    """
    engine = GameEngine(seed=2026, initial_dealer=3)
    human = HumanPlayer(player_id=0, name="Player 1 (Human)")
    mcts1 = MCTSAgent.create_agent_1(player_id=1, simulation_budget=50, seed=11)
    mcts2 = MCTSAgent.create_agent_2(player_id=2, simulation_budget=50, seed=22)
    maxn3 = MaxNAgent(player_id=3, max_depth=2, num_determinizations=2, seed=33)

    players = {0: human, 1: mcts1, 2: mcts2, 3: maxn3}

    # Pre-seed cumulative scores near 16 so the game reaches >= 16 naturally within 1-2 full rounds
    engine._GameEngine__state.cumulative_scores = {0: 14, 1: 14, 2: 14, 3: 14}

    rounds_played = 0
    while not engine.is_game_over and rounds_played < 6:
        if rounds_played > 0:
            engine.start_new_round()
        rounds_played += 1

        # Bidding phase
        for _ in range(4):
            pid = engine.current_player
            obs = engine.get_observation(pid)
            if pid == 0:
                human.set_pending_bid(2)
                bid = human.choose_bid(obs)
            else:
                bid = players[pid].choose_bid(obs)
            assert 0 <= bid <= 13
            engine.submit_bid(pid, bid)

        assert engine.phase == GamePhase.PLAYING

        # All 13 tricks
        for _ in range(13):
            for _ in range(4):
                pid = engine.current_player
                obs = engine.get_observation(pid)
                if pid == 0:
                    # Human chooses highest legal card
                    chosen = max(obs.legal_moves, key=lambda c: c.value)
                    human.set_pending_card(chosen)
                    card = human.choose_card(obs)
                else:
                    card = players[pid].choose_card(obs)
                assert card in obs.legal_moves
                engine.play_card(pid, card)

        assert sum(engine.tricks_won.values()) == 13

    assert engine.is_game_over is True
    assert engine.phase == GamePhase.GAME_OVER
    assert len(engine.winner_ids) >= 1
    for w_id in engine.winner_ids:
        assert engine.cumulative_scores[w_id] >= 16


def test_experimental_evaluation_pipeline() -> None:
    """Verify the automated evaluation benchmark collects all required metrics."""
    factories = {
        0: lambda s: RandomAgent(player_id=0, seed=s),
        1: lambda s: MCTSAgent.create_agent_1(player_id=1, simulation_budget=30, seed=s),
        2: lambda s: MCTSAgent.create_agent_2(player_id=2, simulation_budget=30, seed=s),
        3: lambda s: MaxNAgent(player_id=3, max_depth=2, num_determinizations=2, seed=s),
    }
    report = run_match_experiment(
        agent_factories=factories,
        num_games=1,
        max_rounds_per_game=2,
        base_seed=50,
    )
    assert report.num_games == 1
    assert report.total_rounds >= 1
    table_output = report.format_table()
    assert "Avg Score" in table_output
    assert "Bid Fulfill %" in table_output
    assert "Win Rate %" in table_output


def test_html_css_web_ui_session_and_exact_suit_symbols() -> None:
    """Verify the HTML5/CSS3 WebGameSession controller, static assets, and exact ♠ ♥ ♣ ♦ suit symbols."""
    from ui.config import GameConfig
    from ui.web_ui import UI_DIR, WebGameSession

    for fname in ("index.html", "styles.css", "app.js"):
        assert os.path.exists(os.path.join(UI_DIR, fname)), f"Missing {fname}"

    cfg = GameConfig(mcts_agent1_budget=200, mcts_agent2_budget=300, maxn_search_depth=3)
    session = WebGameSession(cfg)
    state_json = session.get_public_state_json()

    # Verify exact Unicode suit symbols ♠ ♥ ♣ ♦ in serialized cards
    valid_symbols = {"♠", "♥", "♣", "♦"}
    assert len(state_json["human_hand"]) == 13
    for card_obj in state_json["human_hand"]:
        assert card_obj["symbol"] in valid_symbols

    # Verify manual budget and depth update endpoint logic
    session.update_ai_settings(mcts1_budget=1750, mcts2_budget=2250, maxn_depth=5)
    updated_json = session.get_public_state_json()
    assert updated_json["config"]["mcts_agent1_budget"] == 1750
    assert updated_json["config"]["mcts_agent2_budget"] == 2250
    assert updated_json["config"]["maxn_search_depth"] == 5

    # Submit human bid and verify state transition
    res = session.submit_human_bid(3)
    assert res["ok"] is True
    session._cancel_ai()


def test_anticlockwise_bidding_and_starting_player_rotation_across_rounds() -> None:
    """Verify anti-clockwise table rotation (0:Bottom -> 1:Right -> 2:Top -> 3:Left):
    - Round 1: Player 0 (Bottom/Human) bids first and leads Trick 1; order is 0 -> 1 -> 2 -> 3.
    - Round 2: Player 1 (Right/MCTS-1) bids first and leads Trick 1; order is 1 -> 2 -> 3 -> 0 (Human is last).
    """
    engine = GameEngine(seed=101, initial_dealer=3)

    # Round 1: You (0) start -> Right (1) -> Top (2) -> Left (3)
    assert engine.round_number == 1
    assert engine.starting_player == 0
    expected_r1_order = [0, 1, 2, 3]
    for pid in expected_r1_order:
        assert engine.current_player == pid
        engine.submit_bid(pid, 2)
    assert engine.current_player == 0  # Player 0 leads Trick 1

    # Complete all 13 tricks of Round 1
    for _ in range(13):
        for _ in range(4):
            pid = engine.current_player
            engine.play_card(pid, engine.get_legal_moves(pid)[0])

    # Round 2: Right player (1) starts -> Top (2) -> Left (3) -> You (0, last!)
    engine.start_new_round()
    assert engine.round_number == 2
    assert engine.starting_player == 1
    expected_r2_order = [1, 2, 3, 0]
    for pid in expected_r2_order:
        assert engine.current_player == pid
        engine.submit_bid(pid, 2)
    assert engine.current_player == 1  # Right player (1) leads Trick 1 of Round 2


