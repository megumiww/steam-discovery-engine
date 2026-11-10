import unittest
from pathlib import Path

from steam_discovery.baseline import PopularityBaseline
from steam_discovery.data import load_games, load_interactions


ROOT = Path(__file__).parents[1]


class PopularityBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        games = load_games(ROOT / "data/raw/games.csv")
        interactions = load_interactions(ROOT / "data/raw/interactions.csv")
        cls.model = PopularityBaseline(games, interactions)

    def test_excludes_games_seen_by_user(self) -> None:
        recommendations = self.model.recommend("u1", top_k=10)
        ids = {item.game.game_id for item in recommendations}
        self.assertNotIn("g001", ids)
        self.assertNotIn("g004", ids)
        self.assertNotIn("g006", ids)

    def test_returns_sorted_scores(self) -> None:
        recommendations = self.model.recommend("unknown", top_k=5)
        scores = [item.score for item in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
