from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .data import Game
from .tfidf import TfidfRetriever, tokenize


@dataclass(frozen=True)
class BM25Result:
    game: Game
    score: float
    reason: str


class BM25Retriever:
    """A small inverted-index BM25 retriever for game metadata."""

    def __init__(self, games: list[Game], k1: float = 1.5, b: float = 0.75) -> None:
        self.games = games
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.document_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.average_document_length = 0.0
        self._build_index()

    def _build_index(self) -> None:
        document_frequency: Counter[str] = Counter()
        for document_id, game in enumerate(self.games):
            tokens = tokenize(TfidfRetriever._document(game))
            frequencies = Counter(tokens)
            self.document_lengths.append(len(tokens))
            for token, frequency in frequencies.items():
                self.postings[token].append((document_id, frequency))
            document_frequency.update(frequencies.keys())

        document_count = len(self.games)
        self.average_document_length = sum(self.document_lengths) / max(document_count, 1)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score(self, query_terms: list[str], document_id: int) -> float:
        length = self.document_lengths[document_id]
        score = 0.0
        postings_by_term = {term: dict(self.postings.get(term, [])) for term in set(query_terms)}
        for term in query_terms:
            frequency = postings_by_term.get(term, {}).get(document_id, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / max(self.average_document_length, 1)
            )
            score += self.idf.get(term, 0.0) * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        query_terms = tokenize(query)
        candidate_ids = {
            document_id
            for term in query_terms
            for document_id, _ in self.postings.get(term, [])
        }
        ranked = sorted(
            ((document_id, self._score(query_terms, document_id)) for document_id in candidate_ids),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            BM25Result(self.games[document_id], score, f"BM25 lexical match for query: {query}")
            for document_id, score in ranked
            if score > 0
        ]
