from __future__ import annotations

from dataclasses import dataclass

from .baseline import PopularityBaseline
from .bm25 import BM25Retriever
from .data import Game, Interaction
from .tfidf import TfidfRetriever


@dataclass(frozen=True)
class HybridResult:
    game: Game
    score: float
    bm25_score: float
    tfidf_score: float
    popularity_score: float
    reason: str


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 if high > 0 else 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class HybridRanker:
    """Combines lexical retrieval, semantic metadata similarity, and popularity."""

    def __init__(
        self,
        games: list[Game],
        interactions: list[Interaction],
        bm25_weight: float = 0.45,
        tfidf_weight: float = 0.40,
        popularity_weight: float = 0.15,
    ) -> None:
        total = bm25_weight + tfidf_weight + popularity_weight
        if total <= 0:
            raise ValueError("at least one ranking weight must be positive")
        self.games = games
        self.bm25_weight = bm25_weight / total
        self.tfidf_weight = tfidf_weight / total
        self.popularity_weight = popularity_weight / total
        self.bm25 = BM25Retriever(games)
        self.tfidf = TfidfRetriever(games, interactions)
        self.popularity = PopularityBaseline(games, interactions)

    def search(self, query: str, top_k: int = 10) -> list[HybridResult]:
        bm25_results = {item.game.game_id: item.score for item in self.bm25.search(query, len(self.games))}
        tfidf_results = {item.game.game_id: item.score for item in self.tfidf.search(query, len(self.games))}
        popularity_results = {game.game_id: self.popularity.score(game) for game in self.games}
        bm25_norm = _normalize(bm25_results)
        tfidf_norm = _normalize(tfidf_results)
        popularity_norm = _normalize(popularity_results)

        ranked = []
        for game in self.games:
            bm25_score = bm25_norm.get(game.game_id, 0.0)
            tfidf_score = tfidf_norm.get(game.game_id, 0.0)
            popularity_score = popularity_norm.get(game.game_id, 0.0)
            score = (
                self.bm25_weight * bm25_score
                + self.tfidf_weight * tfidf_score
                + self.popularity_weight * popularity_score
            )
            ranked.append(HybridResult(game, score, bm25_score, tfidf_score, popularity_score, "weighted BM25 + TF-IDF + popularity"))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]

    def recommend(self, user_id: str, top_k: int = 10) -> list[HybridResult]:
        content = {item.game.game_id: item.score for item in self.tfidf.recommend(user_id, len(self.games))}
        seen = self.tfidf.seen_by_user.get(user_id, set())
        popularity = {game.game_id: self.popularity.score(game) for game in self.games if game.game_id not in seen}
        content_norm = _normalize(content)
        popularity_norm = _normalize(popularity)
        ranked = []
        for game in self.games:
            if game.game_id in seen:
                continue
            tfidf_score = content_norm.get(game.game_id, 0.0)
            popularity_score = popularity_norm.get(game.game_id, 0.0)
            score = (1 - self.popularity_weight) * tfidf_score + self.popularity_weight * popularity_score
            ranked.append(HybridResult(game, score, 0.0, tfidf_score, popularity_score, "content profile + popularity fallback"))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]
