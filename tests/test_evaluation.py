import unittest

from steam_discovery.evaluation import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_ranking_metrics(self) -> None:
        recommended = ["g1", "g2", "g3"]
        relevant = {"g2", "g4"}
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 3), 1 / 2)
        self.assertEqual(hit_rate_at_k(recommended, relevant, 3), 1.0)
        self.assertGreater(ndcg_at_k(recommended, relevant, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
