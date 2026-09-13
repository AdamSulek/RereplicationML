import unittest

import numpy as np
from rdkit import Chem

from scripts.interpretability.shap_structure_mapping import (
    count_ecfp_and_bit_info,
    environment_atoms_and_bonds,
    select_local_bits,
    verify_fingerprint_row,
)


class ShapStructureMappingTest(unittest.TestCase):
    def test_selection_is_per_compound_and_ignores_absent_bits(self):
        molecule = Chem.MolFromSmiles("CCO")
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        present = sorted(bit_info)
        shap_a = np.zeros(2048)
        shap_b = np.zeros(2048)
        shap_a[present[0]], shap_a[present[1]] = 2.0, 1.0
        shap_b[present[0]], shap_b[present[1]] = 0.1, -3.0
        absent = next(bit for bit in range(2048) if bit not in bit_info)
        shap_a[absent] = 1000.0

        selected_a = select_local_bits(molecule, fingerprint, bit_info, shap_a, top_n=2)
        selected_b = select_local_bits(molecule, fingerprint, bit_info, shap_b, top_n=2)

        self.assertEqual([item.bit_index for item in selected_a], present[:2])
        self.assertEqual([item.bit_index for item in selected_b], [present[1], present[0]])
        self.assertNotIn(absent, [item.bit_index for item in selected_a])

    def test_count_fingerprint_matches_bit_info_occurrences(self):
        molecule = Chem.MolFromSmiles("CCOCC")
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        self.assertEqual(set(np.flatnonzero(fingerprint)), set(bit_info))
        for bit, occurrences in bit_info.items():
            self.assertEqual(int(fingerprint[bit]), len(occurrences))

    def test_radius_two_contains_complete_fragment_and_bonds(self):
        molecule = Chem.MolFromSmiles("CCCCC")
        atoms, bonds = environment_atoms_and_bonds(molecule, center=2, radius=2)
        self.assertEqual(atoms, {0, 1, 2, 3, 4})
        self.assertEqual(bonds, {0, 1, 2, 3})

    def test_verification_rejects_different_fingerprint_row(self):
        molecule = Chem.MolFromSmiles("CCO")
        fingerprint, _ = count_ecfp_and_bit_info(molecule)
        changed = fingerprint.copy()
        changed[np.flatnonzero(changed)[0]] += 1
        with self.assertRaisesRegex(ValueError, "Fingerprint mismatch"):
            verify_fingerprint_row(changed, fingerprint)


if __name__ == "__main__":
    unittest.main()
