"""Abstract base player interface for Call Bridge."""

from __future__ import annotations
from abc import ABC, abstractmethod
from game.card import Card
from game.state import PlayerObservation


class BasePlayer(ABC):
    """Common interface implemented by human and AI players."""

    def __init__(
        self,
        player_id: int,
        name: str,
        agent_type: str = "Base",
        is_human: bool = False,
    ) -> None:
        self.player_id = player_id
        self.name = name
        self.agent_type = agent_type
        self.is_human = is_human

    @abstractmethod
    def choose_bid(self, observation: PlayerObservation) -> int:
        """Select a bid (integer from 0 to 13) given the player's restricted observation."""

    @abstractmethod
    def choose_card(self, observation: PlayerObservation) -> Card:
        """Select a legal card to play given the player's restricted observation."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.player_id}, name={self.name!r}, type={self.agent_type!r})"
