from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Game, Interaction


@dataclass(frozen=True)
class Recommendation:
    game: Game
    score: float
    reason: str


class PopularityBaseline:
    """A transparent baseline for later recommendation experiments."""

    def __init__(self, games: list[Game], interactions: list[Interaction]) -> None:
        self.games = games
        self.seen_by_user: dict[str, set[str]] = {}
        for interaction in interactions:
            self.seen_by_user.setdefault(interaction.user_id, set()).add(interaction.game_id)

        self.global_mean = sum(game.positive_ratio for game in games) / max(len(games), 1)

    def score(self, game: Game) -> float:
        # Bayesian shrinkage prevents a game with very few ratings from winning
        # only because of a small sample size.
        prior_weight = 50
        rating_score = (
            game.rating_count * game.positive_ratio + prior_weight * self.global_mean
        ) / (game.rating_count + prior_weight)
        popularity_signal = math.log1p(game.rating_count) / 20
        return rating_score + popularity_signal

    def _score(self, game: Game) -> float:
        return self.score(game)

    def recommend(self, user_id: str, top_k: int = 10) -> list[Recommendation]:
        seen = self.seen_by_user.get(user_id, set())
        candidates = [game for game in self.games if game.game_id not in seen]
        ranked = sorted(candidates, key=self._score, reverse=True)[:top_k]
        return [
            Recommendation(
                game=game,
                score=self._score(game),
                reason="high Bayesian rating with a popularity prior",
            )
            for game in ranked
        ]
