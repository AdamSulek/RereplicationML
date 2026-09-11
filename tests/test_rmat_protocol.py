import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train import train_rmat


class RMatProtocolTest(unittest.TestCase):
    def test_selection_directions(self):
        self.assertTrue(train_rmat.selection_improved(0.4, 0.5, "val_loss"))
        self.assertFalse(train_rmat.selection_improved(0.6, 0.5, "val_loss"))
        self.assertTrue(train_rmat.selection_improved(0.6, 0.5, "val_roc_auc"))
        self.assertTrue(train_rmat.selection_improved(0.6, 0.5, "val_pr_auc"))

    def test_exact_kfold_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "mcf10a"
            dataset_dir.mkdir()
            for name in [*(f"train_{i}.p" for i in range(10)), "val.p", "test.p"]:
                (dataset_dir / name).touch()
            with patch.object(train_rmat, "DATA_ROOT", root):
                assignment = train_rmat.source_file_assignment("mcf10a", "kfold", 9)
            self.assertEqual([path.name for path in assignment["test"]], ["train_9.p"])
            self.assertEqual([path.name for path in assignment["validation"]], ["train_0.p"])
            self.assertEqual(
                {path.name for path in assignment["train"]},
                {*(f"train_{i}.p" for i in range(1, 9)), "val.p", "test.p"},
            )


if __name__ == "__main__":
    unittest.main()
