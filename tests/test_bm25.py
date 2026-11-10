import unittest
from pathlib import Path

from steam_discovery.bm25 import BM25Retriever
from steam_discovery.data import load_games


ROOT = Path(__file__).parents[1]


class BM25RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.games = load_games(ROOT / "data/raw/games.csv")
        cls.model = BM25Retriever(cls.games)

    def test_search_returns_lexical_matches(self) -> None:
        results = self.model.search("co-op survival", top_k=3)
        titles = [item.game.title for item in results]
        self.assertIn("Ironwood Survival", titles)

    def test_scores_are_sorted(self) -> None:
        results = self.model.search("space exploration", top_k=5)
        scores = [item.score for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_query_returns_no_results(self) -> None:
        self.assertEqual(self.model.search("nonexistentterm", top_k=5), [])


if __name__ == "__main__":
    unittest.main()
