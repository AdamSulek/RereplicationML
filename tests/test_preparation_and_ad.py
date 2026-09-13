import unittest

import numpy as np

from scripts.data_preparation.generate_ecfp import FP_SIZE, configuration, fingerprint_smiles
from scripts.data_preparation.prepare_dataset import activity_labels, canonical_largest_fragment
from scripts.model_evaluation.applicability_domain import FIXED_CUTOFF, applicability_scores


class ChemistryWorkflowTest(unittest.TestCase):
    def test_count_ecfp_is_explicit_integer_2048(self):
        matrix = fingerprint_smiles(["CCO", "CC"])
        self.assertEqual(matrix.shape, (2, FP_SIZE))
        self.assertTrue(np.issubdtype(matrix.dtype, np.integer))
        self.assertTrue(configuration()["count"])
        self.assertEqual(configuration()["radius"], 2)

    def test_largest_fragment_and_labels(self):
        self.assertEqual(canonical_largest_fragment("[Na+].CC(=O)[O-]"), "CC(=O)[O-]")
        labels = activity_labels(
            __import__("pandas").Series(["Active", "inactive"]),
            {"active"},
            {"inactive"},
        )
        self.assertEqual(labels.tolist(), [1, 0])

    def test_applicability_domain_uses_five_neighbors(self):
        row = applicability_scores(["CCO"], ["CCO"] * 5)[0]
        self.assertEqual(row["mean_top5_similarity"], 1.0)
        self.assertEqual(row["nearest_neighbor_similarity"], 1.0)
        self.assertEqual(row["cutoff"], FIXED_CUTOFF)
        self.assertEqual(row["AD_status"], "In-domain")


if __name__ == "__main__":
    unittest.main()
