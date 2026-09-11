#!/usr/bin/env python3
"""Draw top positive ECFP/SHAP environments for XGBoost true positives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.train_xgb import feature_column, index_mask, load_data, model
from scripts.interpretability.shap_structure_mapping import (
    count_ecfp_and_bit_info,
    select_local_bits,
    verify_fingerprint_row,
)


FEATURE_LENGTH = 2048
COLORS = [
    (0.90, 0.20, 0.15),
    (0.95, 0.55, 0.10),
    (0.15, 0.60, 0.25),
    (0.15, 0.40, 0.85),
    (0.60, 0.25, 0.75),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgb_full/tp_shap"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def data_path(dataset: str) -> Path:
    if dataset == "mcf10a":
        return Path("data/mcf10a/mcf10a_xgb_2048.parquet")
    return Path("data/sw480/sw480_clean_xgb.parquet")



def draw_molecule(
    molecule,
    atom_indices: set[int],
    bond_indices: set[int],
    output: Path,
) -> None:
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 760)
    atom_colors = {index: (1.0, 0.65, 0.15) for index in atom_indices}
    bond_colors = {index: (1.0, 0.25, 0.05) for index in bond_indices}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=sorted(atom_indices),
        highlightAtomColors=atom_colors,
        highlightBonds=sorted(bond_indices),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    output.write_text(drawer.GetDrawingText())


def main() -> None:
    args = parse_args()
    namespace = argparse.Namespace(dataset=args.dataset)
    data = load_data(data_path(args.dataset), feature_column(namespace))
    summary_path = Path("artifacts/xgb_full") / args.dataset / "scaffold/scaffold_summary.json"
    params = json.loads(summary_path.read_text())["selected_params"]
    X = data["X"]
    y = data["y"]
    splits = data["split"]
    smiles = data["smiles"]
    train_idx = index_mask(splits, {f"train_{i}" for i in range(10)})
    test_idx = index_mask(splits, {"test"})

    fitted = model(params, y[train_idx], args.n_jobs)
    fitted.fit(X[train_idx], y[train_idx])
    booster = fitted.get_booster()
    probabilities = fitted.predict_proba(X[test_idx])[:, 1]
    contributions = booster.predict(
        xgb.DMatrix(X[test_idx]), pred_contribs=True, validate_features=False
    )[:, :-1]

    true_positive_positions = np.flatnonzero(
        (y[test_idx] == 1) & (probabilities >= args.threshold)
    )
    order = true_positive_positions[np.argsort(-probabilities[true_positive_positions], kind="stable")]
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for tp_rank, position in enumerate(order, start=1):
        row_index = int(test_idx[position])
        molecule_smiles = str(smiles[row_index])
        molecule = Chem.MolFromSmiles(molecule_smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {row_index}: {molecule_smiles}")
        row = X[row_index]
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        verify_fingerprint_row(row.toarray(), fingerprint)
        selected_bits = select_local_bits(
            molecule, fingerprint, bit_info, contributions[position],
            top_n=args.top_n, positive_only=True,
        )
        selected = [(item.bit_index, item.shap_value) for item in selected_bits]

        merged_atoms: set[int] = set()
        merged_bonds: set[int] = set()
        top_bit_rows = []
        for selected_bit in selected_bits:
            bit_rank = selected_bit.rank
            bit = selected_bit.bit_index
            shap_value = selected_bit.shap_value
            bit_atoms: set[int] = set()
            bit_bonds: set[int] = set()
            environment_rows = []
            for occurrence, environment in enumerate(selected_bit.environments, start=1):
                bit_atoms.update(environment.atom_indices)
                bit_bonds.update(environment.bond_indices)
                environment_rows.append(
                    f"{occurrence}:center={environment.center_atom},radius={environment.radius},"
                    f"atoms={chr(44).join(map(str, environment.atom_indices))},"
                    f"bonds={chr(44).join(map(str, environment.bond_indices))}"
                )
            merged_atoms.update(bit_atoms)
            merged_bonds.update(bit_bonds)
            top_bit_rows.append(
                {
                    "bit_rank": bit_rank,
                    "bit_index": bit,
                    "shap_value": shap_value,
                    "fingerprint_count": selected_bit.count,
                    "environment_count": len(selected_bit.environments),
                    "environments": " | ".join(environment_rows),
                    "atom_indices": ";".join(map(str, sorted(bit_atoms))),
                    "bond_indices": ";".join(map(str, sorted(bit_bonds))),
                }
            )

        image_name = f"tp_{tp_rank:04d}_testrow_{row_index}.svg"
        draw_molecule(molecule, merged_atoms, merged_bonds, output / image_name)
        fragment_bonds = [
            bond.GetIdx()
            for bond in molecule.GetBonds()
            if bond.GetBeginAtomIdx() in merged_atoms
            and bond.GetEndAtomIdx() in merged_atoms
        ]
        if merged_atoms:
            try:
                merged_smiles = Chem.MolFragmentToSmiles(
                    molecule,
                    atomsToUse=sorted(merged_atoms),
                    bondsToUse=fragment_bonds,
                    isomericSmiles=True,
                )
            except RuntimeError:
                # The drawing remains valid if disconnected environments cannot
                # be serialized as one fragment SMILES.
                merged_smiles = ""
        else:
            merged_smiles = ""
        summary_rows.append(
            {
                "tp_rank": tp_rank,
                "test_row_index": row_index,
                "smiles": molecule_smiles,
                "predicted_probability": float(probabilities[position]),
                "top_positive_bits": ";".join(str(bit) for bit, _ in selected),
                "top_positive_shap_values": ";".join(f"{value:.8g}" for _, value in selected),
                "merged_atom_count": len(merged_atoms),
                "merged_bond_count": len(merged_bonds),
                "merged_fragment_smiles": merged_smiles,
                "drawing": image_name,
            }
        )
        for bit_row in top_bit_rows:
            records.append(
                {
                    "tp_rank": tp_rank,
                    "test_row_index": row_index,
                    "smiles": molecule_smiles,
                    "predicted_probability": float(probabilities[position]),
                    **bit_row,
                    "drawing": image_name,
                }
            )

    pd.DataFrame(summary_rows).to_csv(output / "true_positive_drawings.csv", index=False)
    pd.DataFrame(records).to_csv(output / "true_positive_top_bits.csv", index=False)
    metadata = {
        "dataset": args.dataset,
        "protocol": "scaffold_selected_model",
        "true_positive_definition": f"true_label == 1 and predicted_probability >= {args.threshold}",
        "top_n_positive_present_bits": args.top_n,
        "test_rows": int(len(test_idx)),
        "true_positive_rows": int(len(order)),
        "feature_length": FEATURE_LENGTH,
        "selected_params": params,
        "output": str(output.resolve()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"dataset: {args.dataset}")
    print(f"scaffold_test_rows: {len(test_idx)}")
    print(f"true_positive_rows: {len(order)}")
    print(f"drawings: {len(order)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
