from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    top = list(recommended)[:k]
    return sum(item in relevant for item in top) / max(len(top), 1)


def recall_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = list(recommended)[:k]
    return sum(item in relevant for item in top) / len(relevant)


def hit_rate_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    return float(any(item in relevant for item in list(recommended)[:k]))


def ndcg_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    top = list(recommended)[:k]
    dcg = sum(1 / math.log2(index + 2) for index, item in enumerate(top) if item in relevant)
    ideal_length = min(len(relevant), k)
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_length))
    return dcg / idcg if idcg else 0.0
