from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .data import Game, Interaction


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ContentRecommendation:
    game: Game
    score: float
    reason: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class TfidfRetriever:
    """TF-IDF retrieval and content-based recommendation over game metadata."""

    def __init__(self, games: list[Game], interactions: list[Interaction]) -> None:
        self.games = games
        self.interactions = interactions
        self.seen_by_user: dict[str, set[str]] = {}
        for interaction in interactions:
            self.seen_by_user.setdefault(interaction.user_id, set()).add(interaction.game_id)
        self.games_by_id = {game.game_id: game for game in games}
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.vectors: dict[str, dict[int, float]] = {}
        self._fit()

    @staticmethod
    def _document(game: Game) -> str:
        # Repeating high-signal fields gives titles and tags more influence
        # than the longer free-form description.
        return " ".join(
            [
                game.title,
                game.title,
                " ".join(game.genres),
                " ".join(game.genres),
                " ".join(game.tags),
                " ".join(game.tags),
                game.description,
            ]
        )

    def _fit(self) -> None:
        documents = [tokenize(self._document(game)) for game in self.games]
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))

        terms = sorted(document_frequency)
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        document_count = len(documents)
        self.idf = [
            math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in terms
        ]

        for game, tokens in zip(self.games, documents):
            counts = Counter(tokens)
            raw_vector = {
                self.vocabulary[token]: (1 + math.log(count)) * self.idf[self.vocabulary[token]]
                for token, count in counts.items()
            }
            self.vectors[game.game_id] = self._normalize(raw_vector)

    @staticmethod
    def _normalize(vector: dict[int, float]) -> dict[int, float]:
        magnitude = math.sqrt(sum(value * value for value in vector.values()))
        if magnitude == 0:
            return {}
        return {index: value / magnitude for index, value in vector.items()}

    @staticmethod
    def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def _query_vector(self, query: str) -> dict[int, float]:
        counts = Counter(tokenize(query))
        raw = {
            self.vocabulary[token]: (1 + math.log(count)) * self.idf[self.vocabulary[token]]
            for token, count in counts.items()
            if token in self.vocabulary
        }
        return self._normalize(raw)

    def search(self, query: str, top_k: int = 10) -> list[ContentRecommendation]:
        query_vector = self._query_vector(query)
        ranked = sorted(
            ((game, self._cosine(query_vector, self.vectors[game.game_id])) for game in self.games),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            ContentRecommendation(game, score, f"metadata similarity for query: {query}")
            for game, score in ranked
            if score > 0
        ]

    def recommend(self, user_id: str, top_k: int = 10) -> list[ContentRecommendation]:
        history = [item for item in self.interactions if item.user_id == user_id]
        seen = {item.game_id for item in history}
        profile: dict[int, float] = {}
        for item in history:
            for index, value in self.vectors.get(item.game_id, {}).items():
                profile[index] = profile.get(index, 0.0) + item.signal * value
        profile = self._normalize(profile)

        candidates = [game for game in self.games if game.game_id not in seen]
        ranked = sorted(
            ((game, self._cosine(profile, self.vectors[game.game_id])) for game in candidates),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            ContentRecommendation(game, score, "similar metadata to the user's history")
            for game, score in ranked
        ]
