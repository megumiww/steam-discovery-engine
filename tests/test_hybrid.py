import unittest
from pathlib import Path

from steam_discovery.data import load_games, load_interactions
from steam_discovery.hybrid import HybridRanker


ROOT = Path(__file__).parents[1]


class HybridRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        games = load_games(ROOT / "data/raw/games.csv")
        interactions = load_interactions(ROOT / "data/raw/interactions.csv")
        cls.model = HybridRanker(games, interactions)

    def test_weights_are_normalized(self) -> None:
        self.assertAlmostEqual(self.model.bm25_weight + self.model.tfidf_weight + self.model.popularity_weight, 1.0)

    def test_search_scores_are_sorted(self) -> None:
        results = self.model.search("co-op survival", top_k=5)
        scores = [item.score for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recommendation_excludes_seen_games(self) -> None:
        results = self.model.recommend("u1", top_k=10)
        ids = {item.game.game_id for item in results}
        self.assertNotIn("g001", ids)
        self.assertNotIn("g004", ids)
        self.assertNotIn("g006", ids)


if __name__ == "__main__":
    unittest.main()
