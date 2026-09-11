import unittest

from scripts.model_evaluation.metrics import screening_metrics


class ScreeningMetricsTest(unittest.TestCase):
    def test_expected_keys_and_enrichment(self):
        result = screening_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        for key in ("roc_auc", "pr_auc", "f1", "recall", "precision", "pr_auc_enrichment", "ef_at_1pct", "ef_at_5pct"):
            self.assertIn(key, result)
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["pr_auc"], 1.0)
        self.assertEqual(result["pr_auc_enrichment"], 2.0)
        self.assertEqual(result["ef_at_1pct"], 2.0)


if __name__ == "__main__":
    unittest.main()
