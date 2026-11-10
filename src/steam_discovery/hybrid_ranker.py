"""Weighted hybrid ranking for popularity, TF-IDF, and BM25 results."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


GameId = Any
Result = Tuple[GameId, float]


class HybridRanker:
    """Normalize and combine three independent retrieval result lists.

    Retriever objects (or callables) may be supplied to the constructor.  A
    retriever is expected to expose ``retrieve(query)``; ``rank(query)`` and
    ``search(query)`` are also accepted for interoperability.
    """

    def __init__(
        self,
        popularity_retriever: Optional[Any] = None,
        tfidf_retriever: Optional[Any] = None,
        bm25_retriever: Optional[Any] = None,
        w1: float = 0.2,
        w2: float = 0.4,
        w3: float = 0.4,
    ) -> None:
        total = float(w1) + float(w2) + float(w3)
        if total == 0.0:
            raise ValueError("retriever weights must not sum to zero")

        self.weights = (float(w1) / total, float(w2) / total, float(w3) / total)
        self.popularity_retriever = popularity_retriever
        self.tfidf_retriever = tfidf_retriever
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def min_max_normalize(results: Iterable[Result]) -> List[Result]:
        """Return scores normalized independently to the interval [0, 1]."""
        items = list(results)
        if not items:
            return []

        scores = [float(score) for _, score in items]
        minimum, maximum = min(scores), max(scores)
        if maximum == minimum:
            normalized = 0.5
            return [(game_id, normalized) for game_id, _ in items]

        span = maximum - minimum
        return [(game_id, (float(score) - minimum) / span) for game_id, score in items]

    def fuse(
        self,
        popularity_results: Iterable[Result],
        tfidf_results: Iterable[Result],
        bm25_results: Iterable[Result],
    ) -> List[Result]:
        """Fuse three result lists and return all game IDs sorted by score."""
        normalized = [
            self.min_max_normalize(popularity_results),
            self.min_max_normalize(tfidf_results),
            self.min_max_normalize(bm25_results),
        ]

        scores_by_route: List[Dict[GameId, float]] = []
        for route in normalized:
            route_scores: Dict[GameId, float] = {}
            for game_id, score in route:
                # If a retriever emits an ID more than once, retain its best score.
                route_scores[game_id] = max(score, route_scores.get(game_id, score))
            scores_by_route.append(route_scores)

        all_game_ids = set()
        for route_scores in scores_by_route:
            all_game_ids.update(route_scores)

        w_pop, w_tfidf, w_bm25 = self.weights
        fused = []
        for game_id in all_game_ids:
            score = (
                w_pop * scores_by_route[0].get(game_id, 0.0)
                + w_tfidf * scores_by_route[1].get(game_id, 0.0)
                + w_bm25 * scores_by_route[2].get(game_id, 0.0)
            )
            fused.append((game_id, score))

        return sorted(fused, key=lambda item: (-item[1], str(item[0])))

    @staticmethod
    def _retrieve(retriever: Any, query: str) -> List[Result]:
        if retriever is None:
            raise ValueError("all three retrievers must be supplied before calling rank()")
        if callable(retriever):
            return list(retriever(query))
        for method_name in ("retrieve", "rank", "search"):
            method: Optional[Callable[[str], Iterable[Result]]] = getattr(
                retriever, method_name, None
            )
            if method is not None:
                return list(method(query))
        raise TypeError("retriever must be callable or expose retrieve(), rank(), or search()")

    def rank(self, query: str, top_k: int = 10) -> List[Result]:
        """Retrieve with all three components, fuse their scores, and return Top-K."""
        if top_k <= 0:
            return []
        results = self.fuse(
            self._retrieve(self.popularity_retriever, query),
            self._retrieve(self.tfidf_retriever, query),
            self._retrieve(self.bm25_retriever, query),
        )
        return results[:top_k]


if __name__ == "__main__":
    popularity = [("game_a", 10.0), ("game_b", 5.0), ("game_c", 5.0)]
    tfidf = [("game_b", 0.2), ("game_c", 0.8), ("game_d", 0.5)]
    bm25 = [("game_a", 1.0), ("game_d", 1.0)]
    ranker = HybridRanker(w1=0.2, w2=0.4, w3=0.4)
    print(ranker.fuse(popularity, tfidf, bm25))
