from __future__ import annotations

import argparse

from .baseline import PopularityBaseline
from .bm25 import BM25Retriever
from .hybrid import HybridRanker
from .data import load_games, load_interactions
from .tfidf import TfidfRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend games with the popularity baseline.")
    parser.add_argument("--games", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("baseline", "content", "bm25", "hybrid"), default="baseline")
    parser.add_argument("--query", help="Text query used by content search mode.")
    args = parser.parse_args()

    games = load_games(args.games)
    interactions = load_interactions(args.interactions)
    if args.mode == "content":
        model = TfidfRetriever(games, interactions)
        results = model.search(args.query, args.top_k) if args.query else model.recommend(args.user, args.top_k)
    elif args.mode == "bm25":
        if not args.query:
            parser.error("--query is required in bm25 mode")
        results = BM25Retriever(games).search(args.query, args.top_k)
    elif args.mode == "hybrid":
        model = HybridRanker(games, interactions)
        results = model.search(args.query, args.top_k) if args.query else model.recommend(args.user, args.top_k)
    else:
        results = PopularityBaseline(games, interactions).recommend(args.user, args.top_k)
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item.game.title} | score={item.score:.4f} | {item.reason}")


if __name__ == "__main__":
    main()
