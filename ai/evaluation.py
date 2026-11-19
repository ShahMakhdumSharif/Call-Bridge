"""Automated experimental evaluation and benchmarking suite for Call Bridge AI agents."""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Optional
from game.engine import GameEngine
from game.rules import NUM_PLAYERS
from players.base_player import BasePlayer
from players.maxn_agent import MaxNAgent
from players.mcts_agent import MCTSAgent
from players.random_agent import RandomAgent


@dataclass
class AgentMetrics:
    """Aggregated evaluation statistics for a single agent across multiple rounds/games."""

    player_id: int
    name: str
    agent_type: str
    games_played: int = 0
    games_won: int = 0
    rounds_played: int = 0
    total_round_score: int = 0
    total_tricks_won: int = 0
    bids_fulfilled: int = 0
    total_decisions: int = 0
    total_decision_time_sec: float = 0.0

    @property
    def average_round_score(self) -> float:
        return self.total_round_score / max(1, self.rounds_played)

    @property
    def average_tricks_won(self) -> float:
        return self.total_tricks_won / max(1, self.rounds_played)

    @property
    def bid_fulfillment_rate(self) -> float:
        return (self.bids_fulfilled / max(1, self.rounds_played)) * 100.0

    @property
    def average_decision_time_ms(self) -> float:
        return (self.total_decision_time_sec / max(1, self.total_decisions)) * 1000.0

    @property
    def win_rate(self) -> float:
        return (self.games_won / max(1, self.games_played)) * 100.0


@dataclass
class ExperimentReport:
    """Complete report of an automated multi-game experiment."""

    title: str
    num_games: int
    total_rounds: int
    seed: int
    metrics: Dict[int, AgentMetrics] = field(default_factory=dict)

    def format_table(self) -> str:
        """Render a clean ASCII/Markdown summary table of all required metrics."""
        lines = [
            f"=== {self.title} (Games: {self.num_games}, Total Rounds: {self.total_rounds}, Seed: {self.seed}) ===",
            f"{'Player / Agent':<28} | {'Avg Score':>9} | {'Avg Tricks':>10} | {'Bid Fulfill %':>13} | {'Avg Time (ms)':>13} | {'Win Rate %':>10}",
            "-" * 96,
        ]
        for pid in sorted(self.metrics.keys()):
            m = self.metrics[pid]
            label = f"{m.name} [{m.agent_type}]"[:28]
            lines.append(
                f"{label:<28} | {m.average_round_score:>9.2f} | {m.average_tricks_won:>10.2f} | "
                f"{m.bid_fulfillment_rate:>12.1f}% | {m.average_decision_time_ms:>13.2f} | {m.win_rate:>9.1f}%"
            )
        lines.append("-" * 96)
        return "\n".join(lines)


def run_match_experiment(
    agent_factories: Dict[int, Callable[[int], BasePlayer]],
    num_games: int = 3,
    max_rounds_per_game: int = 20,
    base_seed: int = 42,
    title: str = "Call Bridge AI Evaluation",
) -> ExperimentReport:
    """Run automated multi-round games between 4 configured AI agents and collect metrics."""
    sample_agents = {pid: factory(base_seed + pid) for pid, factory in agent_factories.items()}
    metrics: Dict[int, AgentMetrics] = {
        pid: AgentMetrics(
            player_id=pid,
            name=sample_agents[pid].name,
            agent_type=sample_agents[pid].agent_type,
        )
        for pid in range(NUM_PLAYERS)
    }

    total_rounds = 0

    for game_idx in range(num_games):
        game_seed = base_seed + game_idx * 100
        engine = GameEngine(seed=game_seed, initial_dealer=(game_idx + 3) % NUM_PLAYERS)
        agents = {pid: factory(game_seed + pid + 1) for pid, factory in agent_factories.items()}

        rounds_in_game = 0
        while not engine.is_game_over and rounds_in_game < max_rounds_per_game:
            if rounds_in_game > 0:
                engine.start_new_round()
            rounds_in_game += 1
            total_rounds += 1

            # Bidding phase
            for _ in range(NUM_PLAYERS):
                pid = engine.current_player
                obs = engine.get_observation(pid)
                t0 = time.perf_counter()
                bid = agents[pid].choose_bid(obs)
                elapsed = time.perf_counter() - t0
                metrics[pid].total_decisions += 1
                metrics[pid].total_decision_time_sec += elapsed
                engine.submit_bid(pid, bid)

            # Playing phase (13 tricks)
            for _ in range(13):
                for _ in range(NUM_PLAYERS):
                    pid = engine.current_player
                    obs = engine.get_observation(pid)
                    t0 = time.perf_counter()
                    card = agents[pid].choose_card(obs)
                    elapsed = time.perf_counter() - t0
                    metrics[pid].total_decisions += 1
                    metrics[pid].total_decision_time_sec += elapsed
                    engine.play_card(pid, card)

            # Collect round statistics
            for pid in range(NUM_PLAYERS):
                b = engine.bids[pid]
                t = engine.tricks_won[pid]
                r_score = engine.round_scores[pid]
                assert b is not None
                metrics[pid].rounds_played += 1
                metrics[pid].total_round_score += r_score
                metrics[pid].total_tricks_won += t
                # Bid is fulfilled if player achieved non-negative score (t == b or t == b + 1)
                if r_score >= 0 and t >= b:
                    metrics[pid].bids_fulfilled += 1

        # Determine game winner(s) (or leading player if max_rounds_per_game reached)
        winners = engine.winner_ids
        if not winners:
            max_sc = max(engine.cumulative_scores.values())
            winners = [pid for pid, sc in engine.cumulative_scores.items() if sc == max_sc]
        for pid in range(NUM_PLAYERS):
            metrics[pid].games_played += 1
            if pid in winners:
                metrics[pid].games_won += 1

    return ExperimentReport(
        title=title,
        num_games=num_games,
        total_rounds=total_rounds,
        seed=base_seed,
        metrics=metrics,
    )


def run_budget_comparison_suite(
    budgets: tuple[int, ...] = (100, 500, 1000),
    num_games: int = 2,
    max_rounds_per_game: int = 10,
    base_seed: int = 100,
) -> List[ExperimentReport]:
    """Compare Random Agent, MCTS Agent 1, MCTS Agent 2, and Max^n Agent across MCTS budgets."""
    reports: List[ExperimentReport] = []
    for budget in budgets:
        factories: Dict[int, Callable[[int], BasePlayer]] = {
            0: lambda s: RandomAgent(player_id=0, name="Player 1 (Random)", seed=s),
            1: lambda s, b=budget: MCTSAgent.create_agent_1(
                player_id=1, name=f"Player 2 (MCTS-1@{b})", simulation_budget=b, seed=s
            ),
            2: lambda s, b=budget: MCTSAgent.create_agent_2(
                player_id=2, name=f"Player 3 (MCTS-2@{b})", simulation_budget=b, seed=s
            ),
            3: lambda s: MaxNAgent(
                player_id=3, name="Player 4 (Max^n)", max_depth=4, num_determinizations=4, seed=s
            ),
        }
        report = run_match_experiment(
            agent_factories=factories,
            num_games=num_games,
            max_rounds_per_game=max_rounds_per_game,
            base_seed=base_seed,
            title=f"Benchmark Suite (MCTS Budget = {budget} simulations)",
        )
        reports.append(report)
    return reports


if __name__ == "__main__":
    suite_reports = run_budget_comparison_suite(budgets=(100, 500), num_games=2, max_rounds_per_game=8)
    for rep in suite_reports:
        print(rep.format_table())
        print()
