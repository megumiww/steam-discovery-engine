"""Evaluation utilities for Steam recommendation models."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from . import bm25_retriever, data, hybrid_ranker, popularity_baseline, tfidf_retriever


def _load_interactions() -> Any:
    """Load interactions through :mod:`data` without touching raw files directly."""
    for name in ("load_interactions", "get_interactions", "load_data"):
        loader = getattr(data, name, None)
        if loader is None:
            continue
        loaded = loader()
        if isinstance(loaded, Mapping):
            if "interactions" in loaded:
                return loaded["interactions"]
            if "games" in loaded and "interactions" not in loaded:
                continue
        if isinstance(loaded, tuple) and len(loaded) >= 2:
            # Common convention: (games, interactions).
            return loaded[1]
        return loaded
    raise AttributeError("data.py must expose an interaction loading function")


def _load_games() -> Any:
    for name in ("load_games", "get_games", "load_data"):
        loader = getattr(data, name, None)
        if loader is None:
            continue
        loaded = loader()
        if isinstance(loaded, Mapping):
            if "games" in loaded:
                return loaded["games"]
            continue
        if isinstance(loaded, tuple) and len(loaded) >= 2:
            return loaded[0]
        return loaded
    return None


def _field(row: Any, *names: str) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
    else:
        for name in names:
            if hasattr(row, name):
                return getattr(row, name)
    return None


def build_ground_truth(interactions: Iterable[Any]) -> Dict[Any, Set[Any]]:
    """Build ``user_id -> set(game_id)`` from loaded interaction records."""
    truth: Dict[Any, Set[Any]] = defaultdict(set)
    if isinstance(interactions, Mapping):
        for user_id, games in interactions.items():
            if isinstance(games, (str, bytes)):
                games = [games]
            for game_id in games:
                truth[user_id].add(game_id)
        return dict(truth)
    for row in interactions:
        user_id = _field(row, "user_id", "userid", "user", "uid")
        game_id = _field(row, "game_id", "gameid", "appid", "item_id", "item")
        if user_id is not None and game_id is not None:
            truth[user_id].add(game_id)
    return dict(truth)


def precision_at_k(recommended: Sequence[Any], ground_truth: Set[Any], k: int = 10) -> float:
    top = list(recommended[:k])
    return sum(1 for game_id in top if game_id in ground_truth) / float(k or 1)


def recall_at_k(recommended: Sequence[Any], ground_truth: Set[Any], k: int = 10) -> float:
    if not ground_truth:
        return 0.0
    return sum(1 for game_id in recommended[:k] if game_id in ground_truth) / float(len(ground_truth))


def ndcg_at_k(recommended: Sequence[Any], ground_truth: Set[Any], k: int = 10) -> float:
    top = list(recommended[:k])
    dcg = sum((1.0 / math.log2(rank + 2)) for rank, game_id in enumerate(top) if game_id in ground_truth)
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _call_model(module: Any, query: str, k: int, games: Any = None, interactions: Any = None, **kwargs: Any) -> List[Any]:
    """Call a retriever while accommodating the small public API variations."""
    candidates = ("recommend", "retrieve", "search", "rank", "get_recommendations")
    for name in candidates:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        for call in (
            lambda: fn(query, k=k, **kwargs),
            lambda: fn(query, top_k=k, **kwargs),
            lambda: fn(games, query, k=k, **kwargs),
            lambda: fn(games, query, top_k=k, **kwargs),
            lambda: fn(games, interactions, query, k=k, **kwargs),
            lambda: fn(games, interactions, query, top_k=k, **kwargs),
            lambda: fn(query, **kwargs),
            lambda: fn(k=k, **kwargs),
        ):
            try:
                result = call()
                return [item[0] if isinstance(item, (tuple, list)) and item else item for item in result][:k]
            except TypeError:
                continue
    raise AttributeError("retriever module exposes no supported recommendation function")


def evaluate(k: int = 10, hybrid_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4)) -> Dict[str, Dict[str, float]]:
    interactions = _load_interactions()
    games = _load_games()
    ground_truth = build_ground_truth(interactions)
    results: Dict[str, List[float]] = {}
    model_modules = {
        "Popularity": popularity_baseline,
        "TF-IDF": tfidf_retriever,
        "BM25": bm25_retriever,
        "Hybrid": hybrid_ranker,
    }
    for model_name, module in model_modules.items():
        values = [0.0, 0.0, 0.0]
        for user_id, truth in ground_truth.items():
            query = str(next(iter(truth), ""))
            kwargs = {"weights": hybrid_weights} if model_name == "Hybrid" else {}
            recommended = _call_model(module, query, k, games=games, interactions=interactions, **kwargs)
            values[0] += precision_at_k(recommended, truth, k)
            values[1] += recall_at_k(recommended, truth, k)
            values[2] += ndcg_at_k(recommended, truth, k)
        count = float(len(ground_truth) or 1)
        results[model_name] = {
            "precision": values[0] / count,
            "recall": values[1] / count,
            "ndcg": values[2] / count,
        }
    return results


def print_comparison(results: Mapping[str, Mapping[str, float]], k: int = 10) -> None:
    print(f"Model | Precision@{k} | Recall@{k} | NDCG@{k}")
    for name, metrics in results.items():
        print(f"{name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['ndcg']:.3f}")


if __name__ == "__main__":
    print_comparison(evaluate(k=10), k=10)
