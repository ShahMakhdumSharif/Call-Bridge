"""HTML5 + CSS3 Web Interface Server for the Call Bridge Python Game Engine & AI Agents.

Uses Python's built-in `http.server` (zero external dependencies) to serve
`ui/index.html`, `ui/styles.css`, and `ui/app.js` and connect them to the
central `GameEngine`, `MCTSAgent` (1 & 2), and `MaxNAgent` (3).
"""

from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import webbrowser
from ai.bidding import estimate_hand_tricks
from game.card import Card, Suit
from game.engine import GameEngine
from game.state import GamePhase
from players.base_player import BasePlayer
from players.human_player import HumanPlayer
from players.maxn_agent import MaxNAgent
from players.mcts_agent import MCTSAgent
from ui.config import GameConfig

UI_DIR = os.path.dirname(os.path.abspath(__file__))


class WebGameSession:
    """Thread-safe controller bridging the Python GameEngine & AI agents to the HTML/CSS frontend."""

    def __init__(self, config: GameConfig) -> None:
        self.lock = threading.Lock()
        self.config = config
        self.engine = GameEngine(initial_dealer=3)
        self.human_player = HumanPlayer(player_id=0, name="Player 1 (You)")
        self.players: List[BasePlayer] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ai_future: Optional[Future[Any]] = None
        self._ai_task_type: Optional[str] = None
        self._ai_acting_pid: Optional[int] = None

        self.displayed_trick: List[Dict[str, Any]] = []
        self.last_trick_winner: Optional[int] = None
        self.last_trick_winner_name: str = ""
        self.last_trick_winning_card: str = ""
        self.trick_hold_until: float = 0.0
        self.status_message: str = ""
        self.error_message: str = ""

        self.start_new_game()

    def _cancel_ai(self) -> None:
        if self._ai_future is not None:
            self._ai_future.cancel()
        self._ai_future = None
        self._ai_task_type = None
        self._ai_acting_pid = None

    def start_new_game(self) -> None:
        with self.lock:
            self._cancel_ai()
            self.human_player = HumanPlayer(player_id=0, name="Player 1 (You)")
            agent1 = MCTSAgent.create_agent_1(
                player_id=1,
                name="Player 2 (MCTS-1)",
                simulation_budget=self.config.mcts_agent1_budget,
            )
            agent2 = MCTSAgent.create_agent_2(
                player_id=2,
                name="Player 3 (MCTS-2)",
                simulation_budget=self.config.mcts_agent2_budget,
            )
            agent3 = MaxNAgent(
                player_id=3,
                name="Player 4 (Max^n)",
                max_depth=self.config.maxn_search_depth,
            )
            self.players = [self.human_player, agent1, agent2, agent3]
            self.engine.start_new_game(initial_dealer=3)
            self._reset_round_view()

    def start_next_round(self) -> None:
        with self.lock:
            if self.engine.phase != GamePhase.ROUND_OVER:
                return
            self._cancel_ai()
            self.engine.start_new_round()
            self._reset_round_view()

    def update_ai_settings(self, mcts1_budget: int, mcts2_budget: int, maxn_depth: int) -> None:
        with self.lock:
            b1 = max(10, min(50000, int(mcts1_budget)))
            b2 = max(10, min(50000, int(mcts2_budget)))
            d3 = max(1, min(12, int(maxn_depth)))
            self.config.mcts_agent1_budget = b1
            self.config.mcts_agent2_budget = b2
            self.config.maxn_search_depth = d3
            if len(self.players) > 1 and isinstance(self.players[1], MCTSAgent):
                self.players[1].simulation_budget = b1
            if len(self.players) > 2 and isinstance(self.players[2], MCTSAgent):
                self.players[2].simulation_budget = b2
            if len(self.players) > 3 and isinstance(self.players[3], MaxNAgent):
                self.players[3].max_depth = d3

    def _reset_round_view(self) -> None:
        self.displayed_trick = []
        self.last_trick_winner = None
        self.last_trick_winner_name = ""
        self.last_trick_winning_card = ""
        self.trick_hold_until = 0.0
        self.error_message = ""
        self._update_status_text()

    @staticmethod
    def _serialize_card(card: Card, is_legal: bool = True) -> Dict[str, Any]:
        return {
            "code": card.ascii_code,
            "rank": card.rank.label,
            "suit": card.suit.value,
            "symbol": card.suit.symbol,  # Exact ♠ ♥ ♣ ♦
            "is_red": card.suit.is_red,
            "is_trump": card.is_trump,
            "value": card.value,
            "is_legal": is_legal,
        }

    def submit_human_bid(self, bid: int) -> Dict[str, Any]:
        with self.lock:
            if self.engine.phase != GamePhase.BIDDING or self.engine.current_player != 0:
                return {"ok": False, "error": "It is not your turn to bid."}
            try:
                self.human_player.set_pending_bid(int(bid))
                obs = self.engine.get_observation(0)
                chosen_bid = self.human_player.choose_bid(obs)
                self.engine.submit_bid(0, chosen_bid)
                self.error_message = ""
                self._update_status_text()
                return {"ok": True}
            except Exception as exc:
                self.error_message = str(exc)
                return {"ok": False, "error": str(exc)}

    def play_human_card(self, card_code: str) -> Dict[str, Any]:
        with self.lock:
            now = time.time()
            if now < self.trick_hold_until:
                return {"ok": False, "error": "Please wait for the completed trick to clear."}
            if self.engine.phase != GamePhase.PLAYING or self.engine.current_player != 0:
                return {"ok": False, "error": "It is not your turn to play a card."}

            try:
                card = Card.from_str(card_code)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

            obs = self.engine.get_observation(0)
            if card not in obs.legal_moves:
                lead = obs.lead_suit
                if lead is not None:
                    self.error_message = f"Illegal Move: You must follow the lead suit ({lead.value} {lead.symbol})!"
                else:
                    self.error_message = "Illegal card selection."
                return {"ok": False, "error": self.error_message}

            self.human_player.set_pending_card(card)
            chosen = self.human_player.choose_card(obs)
            self.error_message = ""
            self._apply_card_play_locked(0, chosen)
            return {"ok": True}

    def _apply_card_play_locked(self, pid: int, card: Card) -> None:
        trick_before = [
            {"player_id": p, "card": self._serialize_card(c)}
            for p, c in self.engine.current_trick
        ]
        trick_before.append({"player_id": pid, "card": self._serialize_card(card)})

        winner_id = self.engine.play_card(pid, card)
        if winner_id is not None:
            self.displayed_trick = trick_before
            self.last_trick_winner = winner_id
            self.last_trick_winner_name = self.players[winner_id].name
            win_entry = next(item for item in trick_before if item["player_id"] == winner_id)
            self.last_trick_winning_card = f"{win_entry['card']['rank']}{win_entry['card']['symbol']}"
            self.trick_hold_until = time.time() + 1.15
            self.status_message = (
                f"{self.last_trick_winner_name} wins Trick #{len(self.engine.trick_history)} "
                f"with {self.last_trick_winning_card}!"
            )
        else:
            self.displayed_trick = [
                {"player_id": p, "card": self._serialize_card(c)}
                for p, c in self.engine.current_trick
            ]
            self.last_trick_winner = None
            self._update_status_text()

    def _update_status_text(self) -> None:
        phase = self.engine.phase
        curr = self.engine.current_player
        p_name = self.players[curr].name

        if phase == GamePhase.BIDDING:
            if curr == 0:
                self.status_message = "Your Turn to Bid — Select your expected tricks (0–13) below."
            else:
                self.status_message = f"Bidding Phase: {p_name} is calculating an independent bid..."
        elif phase == GamePhase.PLAYING:
            if curr == 0:
                lead = self.engine.lead_suit
                if lead is None:
                    self.status_message = "Your Turn — You have the lead! Click any card in your hand to play."
                else:
                    obs = self.engine.get_observation(0)
                    has_lead = any(c.suit == lead for c in obs.hand)
                    if has_lead:
                        self.status_message = f"Your Turn — Follow the lead suit: {lead.value} ({lead.symbol})."
                    else:
                        self.status_message = f"Your Turn — Void in {lead.value} ({lead.symbol})! You may trump with Spades (♠) or play any card."
            else:
                self.status_message = f"{p_name} is thinking..."
        elif phase == GamePhase.ROUND_OVER:
            self.status_message = f"Round {self.engine.round_number} Complete! Review round results."
        elif phase == GamePhase.GAME_OVER:
            self.status_message = "Game Over! Winning condition (>= 16 cumulative points) reached."

    def step_ai_if_needed(self) -> None:
        with self.lock:
            now = time.time()
            if self.trick_hold_until > 0 and now >= self.trick_hold_until:
                self.trick_hold_until = 0.0
                self.displayed_trick = [
                    {"player_id": p, "card": self._serialize_card(c)}
                    for p, c in self.engine.current_trick
                ]
                self.last_trick_winner = None
                self._update_status_text()

            if now < self.trick_hold_until:
                return

            phase = self.engine.phase
            if phase not in (GamePhase.BIDDING, GamePhase.PLAYING):
                return

            curr_pid = self.engine.current_player
            curr_player = self.players[curr_pid]
            if curr_player.is_human:
                return

            if self._ai_future is not None:
                if not self._ai_future.done():
                    return
                result = self._ai_future.result()
                task_type = self._ai_task_type
                acting_pid = self._ai_acting_pid
                self._ai_future = None
                self._ai_task_type = None
                self._ai_acting_pid = None

                if acting_pid != curr_pid:
                    return

                if task_type == "BID" and isinstance(result, int):
                    self.engine.submit_bid(curr_pid, result)
                    self._update_status_text()
                elif task_type == "CARD" and isinstance(result, Card):
                    self._apply_card_play_locked(curr_pid, result)
                return

            obs = self.engine.get_observation(curr_pid)
            self._ai_acting_pid = curr_pid
            if phase == GamePhase.BIDDING:
                self._ai_task_type = "BID"
                self._ai_future = self._executor.submit(curr_player.choose_bid, obs)
            elif phase == GamePhase.PLAYING:
                self._ai_task_type = "CARD"
                self._ai_future = self._executor.submit(curr_player.choose_card, obs)

    def get_public_state_json(self) -> Dict[str, Any]:
        self.step_ai_if_needed()
        with self.lock:
            now = time.time()
            holding_trick = now < self.trick_hold_until
            obs0 = self.engine.get_observation(0)
            legal_set = set(obs0.legal_moves)
            is_human_play_turn = (
                self.engine.phase == GamePhase.PLAYING
                and self.engine.current_player == 0
                and not holding_trick
            )

            human_hand = [
                self._serialize_card(c, is_legal=(c in legal_set) if is_human_play_turn else True)
                for c in obs0.hand
            ]

            est_tricks = round(estimate_hand_tricks(obs0.hand), 1)
            suggested_bid = max(0, min(13, int(round(est_tricks - 0.15))))
            spade_count = sum(1 for c in obs0.hand if c.suit == Suit.SPADES)
            honor_count = sum(1 for c in obs0.hand if c.value >= 12)

            players_info = []
            for pid, p in enumerate(self.players):
                ai_meta = "Human Player"
                if isinstance(p, MCTSAgent):
                    ai_meta = f"{p.agent_type} • {p.simulation_budget} sims"
                elif isinstance(p, MaxNAgent):
                    ai_meta = f"{p.agent_type} • Depth {p.max_depth}"

                players_info.append({
                    "id": pid,
                    "name": p.name,
                    "agent_type": p.agent_type,
                    "ai_meta": ai_meta,
                    "bid": self.engine.bids[pid],
                    "tricks_won": self.engine.tricks_won[pid],
                    "round_score": self.engine.round_scores[pid],
                    "cumulative_score": self.engine.cumulative_scores[pid],
                    "remaining_cards": obs0.remaining_card_counts[pid],
                    "is_dealer": self.engine.dealer == pid,
                    "is_starting_player": self.engine.starting_player == pid,
                    "is_turn": (
                        self.engine.phase in (GamePhase.BIDDING, GamePhase.PLAYING)
                        and self.engine.current_player == pid
                        and not holding_trick
                    ),
                    "is_trick_winner": self.last_trick_winner == pid and holding_trick,
                })

            lead_suit_info = None
            if self.engine.lead_suit is not None:
                lead_suit_info = {
                    "name": self.engine.lead_suit.value,
                    "symbol": self.engine.lead_suit.symbol,
                }

            return {
                "phase": self.engine.phase.value if not holding_trick else "PLAYING",
                "real_phase": self.engine.phase.value,
                "holding_trick": holding_trick,
                "round_number": self.engine.round_number,
                "trick_number": min(13, self.engine.trick_number),
                "winning_score": self.engine.winning_score,
                "dealer": self.engine.dealer,
                "starting_player": self.engine.starting_player,
                "current_player": self.engine.current_player,
                "lead_suit": lead_suit_info,
                "status_message": self.status_message,
                "error_message": self.error_message,
                "players": players_info,
                "human_hand": human_hand,
                "hand_stats": {
                    "estimated_tricks": est_tricks,
                    "suggested_bid": suggested_bid,
                    "spade_count": spade_count,
                    "honor_count": honor_count,
                },
                "displayed_trick": self.displayed_trick,
                "last_trick_winner": self.last_trick_winner if holding_trick else None,
                "winner_ids": self.engine.winner_ids,
                "config": {
                    "mcts_agent1_budget": self.config.mcts_agent1_budget,
                    "mcts_agent2_budget": self.config.mcts_agent2_budget,
                    "maxn_search_depth": self.config.maxn_search_depth,
                },
            }


def run_web_ui(config: GameConfig, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the local HTML5/CSS3 Call Bridge server and open it in the user's browser."""
    session = WebGameSession(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self._send_bytes(raw, "application/json; charset=utf-8", status=status)

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index.html"):
                with open(os.path.join(UI_DIR, "index.html"), "rb") as f:
                    self._send_bytes(f.read(), "text/html; charset=utf-8")
                return
            if self.path.startswith("/styles.css"):
                with open(os.path.join(UI_DIR, "styles.css"), "rb") as f:
                    self._send_bytes(f.read(), "text/css; charset=utf-8")
                return
            if self.path.startswith("/app.js"):
                with open(os.path.join(UI_DIR, "app.js"), "rb") as f:
                    self._send_bytes(f.read(), "application/javascript; charset=utf-8")
                return
            if self.path.startswith("/api/state"):
                self._send_json(session.get_public_state_json())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}

            if self.path == "/api/bid":
                self._send_json(session.submit_human_bid(int(data.get("bid", 0))))
                return
            if self.path == "/api/play":
                self._send_json(session.play_human_card(str(data.get("card", ""))))
                return
            if self.path == "/api/next_round":
                session.start_next_round()
                self._send_json({"ok": True})
                return
            if self.path == "/api/new_game":
                session.start_new_game()
                self._send_json({"ok": True})
                return
            if self.path == "/api/settings":
                session.update_ai_settings(
                    mcts1_budget=int(data.get("mcts1_budget", 500)),
                    mcts2_budget=int(data.get("mcts2_budget", 500)),
                    maxn_depth=int(data.get("maxn_depth", 4)),
                )
                self._send_json({"ok": True})
                return
            self.send_error(404)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"\n==============================================================")
    print(f"  Call Bridge HTML5/CSS3 UI running at: {url}")
    print(f"  Press Ctrl+C in the terminal to stop the server.")
    print(f"==============================================================\n")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Call Bridge web server...")
    finally:
        server.server_close()
