#!/usr/bin/env python3
"""Create auditable per-compound SHAP-to-ECFP structure examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.interpretability.shap_structure_mapping import count_ecfp_and_bit_info, select_local_bits


POSITIVE = (1.0, 0.55, 0.10)
NEGATIVE = (0.20, 0.55, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], required=True)
    parser.add_argument("--shap-dir", type=Path, default=Path("artifacts/xgb_full/shap"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgb_full/shap_structure_examples"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def draw(molecule: Chem.Mol, selected, output: Path, legend: str) -> None:
    atom_colors = {}
    bond_colors = {}
    center_labels: dict[int, list[str]] = {}
    for bit in reversed(selected):
        color = POSITIVE if bit.shap_value >= 0 else NEGATIVE
        for environment in bit.environments:
            for atom in environment.atom_indices:
                atom_colors[atom] = color
            for bond in environment.bond_indices:
                bond_colors[bond] = color
            center_labels.setdefault(environment.center_atom, []).append(f"b{bit.rank}/R{environment.radius}")

    display = Chem.Mol(molecule)
    for atom_index, labels in center_labels.items():
        display.GetAtomWithIdx(atom_index).SetProp("atomNote", ",".join(labels))
    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 850)
    options = drawer.drawOptions()
    options.fillHighlights = True
    options.continuousHighlight = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        display,
        legend=legend,
        highlightAtoms=sorted(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=sorted(bond_colors),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    output.write_text(drawer.GetDrawingText())

    png_drawer = rdMolDraw2D.MolDraw2DCairo(1200, 850)
    png_options = png_drawer.drawOptions()
    png_options.fillHighlights = True
    png_options.continuousHighlight = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        png_drawer,
        display,
        legend=legend,
        highlightAtoms=sorted(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=sorted(bond_colors),
        highlightBondColors=bond_colors,
    )
    png_drawer.FinishDrawing()
    output.with_suffix(".png").write_bytes(png_drawer.GetDrawingText())


def fragment_smiles(molecule: Chem.Mol, atoms: tuple[int, ...], bonds: tuple[int, ...]) -> str:
    return Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=list(atoms),
        bondsToUse=list(bonds),
        isomericSmiles=True,
    )


def main() -> None:
    args = parse_args()
    base_path = args.shap_dir / args.dataset / "external_shap_base.csv"
    with base_path.open(newline="") as handle:
        compounds = list(csv.DictReader(handle))
    shap_path = args.shap_dir / args.dataset / "external_shap_values.npz"
    shap_values = np.load(shap_path)["values"]
    if len(compounds) != shap_values.shape[0]:
        raise ValueError(f"Compound/SHAP row mismatch: {len(compounds)} vs {shap_values.shape[0]}")

    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    bit_rows = []
    environment_rows = []
    for row_index in args.indices:
        compound = compounds[row_index]
        molecule = Chem.MolFromSmiles(compound["smiles"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES in external row {row_index}")
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        selected = select_local_bits(
            molecule, fingerprint, bit_info, shap_values[row_index], top_n=args.top_n
        )
        image_name = f"row_{row_index:03d}_{compound['compound_id']}.svg"
        legend = f"{args.dataset}; row={row_index}; compound={compound['compound_id']}; orange=SHAP+, blue=SHAP-"
        draw(molecule, selected, output / image_name, legend)

        for bit in selected:
            bit_rows.append(
                {
                    "row_index": row_index,
                    "compound_id": compound["compound_id"],
                    "smiles": compound["smiles"],
                    "bit_rank": bit.rank,
                    "bit_index": bit.bit_index,
                    "shap_value": bit.shap_value,
                    "fingerprint_count": bit.count,
                    "environment_count": len(bit.environments),
                    "drawing": image_name,
                }
            )
            for occurrence, environment in enumerate(bit.environments, start=1):
                environment_rows.append(
                    {
                        "row_index": row_index,
                        "compound_id": compound["compound_id"],
                        "bit_rank": bit.rank,
                        "bit_index": bit.bit_index,
                        "shap_value": bit.shap_value,
                        "occurrence": occurrence,
                        "center_atom": environment.center_atom,
                        "radius": environment.radius,
                        "atom_indices": ";".join(map(str, environment.atom_indices)),
                        "bond_indices": ";".join(map(str, environment.bond_indices)),
                        "environment_smiles": fragment_smiles(
                            molecule, environment.atom_indices, environment.bond_indices
                        ),
                        "drawing": image_name,
                    }
                )

    for name, rows in (("selected_bits.csv", bit_rows), ("environments.csv", environment_rows)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "dataset": args.dataset,
        "selection": "top absolute SHAP among nonzero bits in the same compound",
        "global_mean_abs_shap_used": False,
        "indices": args.indices,
        "top_n": args.top_n,
        "shap_values": str(shap_path.resolve()),
        "compound_rows": str(base_path.resolve()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(args.indices)} examples to {output}")


if __name__ == "__main__":
    main()
