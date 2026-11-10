import unittest
from pathlib import Path

from steam_discovery.data import load_games, load_interactions
from steam_discovery.tfidf import TfidfRetriever, tokenize


ROOT = Path(__file__).parents[1]


class TfidfRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.games = load_games(ROOT / "data/raw/games.csv")
        cls.interactions = load_interactions(ROOT / "data/raw/interactions.csv")
        cls.model = TfidfRetriever(cls.games, cls.interactions)

    def test_tokenizer_is_case_insensitive(self) -> None:
        self.assertEqual(tokenize("Co-op Survival!"), ["co", "op", "survival"])

    def test_search_returns_matching_metadata(self) -> None:
        results = self.model.search("co-op survival", top_k=3)
        titles = [item.game.title for item in results]
        self.assertIn("Ironwood Survival", titles)

    def test_recommendation_excludes_seen_games(self) -> None:
        results = self.model.recommend("u1", top_k=10)
        ids = {item.game.game_id for item in results}
        self.assertNotIn("g001", ids)
        self.assertNotIn("g004", ids)
        self.assertNotIn("g006", ids)

    def test_scores_are_sorted(self) -> None:
        results = self.model.search("space exploration", top_k=5)
        scores = [item.score for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
