#!/usr/bin/env python3
"""Summarize local positive-SHAP environments for scaffold true positives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.interpretability.shap_structure_mapping import count_ecfp_and_bit_info, environment_atoms_and_bonds


PATTERNS = [
    ("nitro group", "[N+](=O)[O-]"),
    ("sulfonyl group", "S(=O)(=O)"),
    ("phosphoryl/phosphate group", "P(=O)"),
    ("carboxylic acid/carboxylate", "C(=O)[O;H1,-1]"),
    ("amide/urea/lactam carbonyl", "C(=O)N"),
    ("ester/carbonate carbonyl", "C(=O)O[#6]"),
    ("nitrile", "C#N"),
    ("carbonyl group", "[CX3]=[OX1]"),
    ("N-N motif", "[N,n]-[N,n]"),
]
COMPILED_PATTERNS = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in PATTERNS]


def contains_pattern(molecule: Chem.Mol, atom_indices: set[int], pattern: Chem.Mol) -> bool:
    return any(set(match) <= atom_indices for match in molecule.GetSubstructMatches(pattern))


def motif_class(molecule: Chem.Mol, atom_indices: tuple[int, ...], radius: int) -> str:
    atoms = [molecule.GetAtomWithIdx(index) for index in atom_indices]
    atom_set = set(atom_indices)
    if radius == 0:
        atom = atoms[0]
        symbol = atom.GetSymbol()
        if atom.GetIsAromatic():
            if symbol == "C":
                return "aromatic carbon atom"
            if symbol == "N" and atom.GetTotalNumHs() > 0:
                return "pyrrolic [nH] atom"
            if symbol == "S":
                return "aromatic sulfur atom"
            return "aromatic heteroatom"
        if symbol in {"F", "Cl", "Br", "I"}:
            return "halogen atom"
        if symbol == "O":
            if any(bond.GetBondTypeAsDouble() == 2 for bond in atom.GetBonds()):
                return "carbonyl oxygen atom"
            return "oxygen atom"
        if symbol == "N":
            return "non-aromatic nitrogen atom"
        if symbol == "C":
            return "aliphatic carbon atom"
        return f"{symbol} atom"

    for name, pattern in COMPILED_PATTERNS:
        if contains_pattern(molecule, atom_set, pattern):
            return name

    aromatic_atoms = [atom for atom in atoms if atom.GetIsAromatic()]
    contains_halogen = any(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms)
    if any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0 for atom in atoms) and any(
        atom.GetAtomicNum() == 8 for atom in atoms
    ):
        return "N-oxide/nitro local environment"
    if contains_halogen and aromatic_atoms:
        return "halogenated aryl environment"
    if len(aromatic_atoms) >= 3:
        if any(atom.GetAtomicNum() not in {6, 1} for atom in aromatic_atoms):
            return "heteroaromatic ring environment"
        return "carbocyclic aryl environment"

    internal_bonds = [
        molecule.GetBondWithIdx(index)
        for index in range(molecule.GetNumBonds())
        if molecule.GetBondWithIdx(index).GetBeginAtomIdx() in atom_set
        and molecule.GetBondWithIdx(index).GetEndAtomIdx() in atom_set
    ]
    if any(
        bond.GetBondTypeAsDouble() == 1
        and {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {6, 8}
        for bond in internal_bonds
    ):
        return "ether/alcohol C-O environment"
    if any(atom.GetAtomicNum() == 7 for atom in atoms):
        return "non-aromatic nitrogen environment"
    if all(atom.GetAtomicNum() in {1, 6} for atom in atoms):
        if any(bond.GetBondTypeAsDouble() > 1 for bond in internal_bonds):
            return "unsaturated carbon environment"
        return "aliphatic hydrocarbon environment"
    if contains_halogen:
        return "halogenated aliphatic environment"
    return "other heteroatom environment"


def fragment_smiles(molecule: Chem.Mol, atoms: tuple[int, ...], bonds: tuple[int, ...]) -> str:
    return Chem.MolFragmentToSmiles(
        molecule, atomsToUse=list(atoms), bondsToUse=list(bonds), isomericSmiles=True
    )


def draw_clean(molecule: Chem.Mol, atoms: set[int], bonds: set[int], output: Path) -> None:
    atom_colors = {index: (1.0, 0.55, 0.10) for index in atoms}
    bond_colors = {index: (1.0, 0.30, 0.05) for index in bonds}
    for drawer, target in (
        (rdMolDraw2D.MolDraw2DSVG(1100, 780), output.with_suffix(".svg")),
        (rdMolDraw2D.MolDraw2DCairo(1100, 780), output.with_suffix(".png")),
    ):
        drawer.drawOptions().fillHighlights = True
        drawer.drawOptions().continuousHighlight = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            molecule,
            highlightAtoms=sorted(atoms),
            highlightAtomColors=atom_colors,
            highlightBonds=sorted(bonds),
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
        if target.suffix == ".svg":
            target.write_text(drawing)
        else:
            target.write_bytes(drawing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], default="mcf10a")
    parser.add_argument("--tp-dir", type=Path, default=Path("artifacts/xgb_full/tp_shap"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgb_full/tp_motif_analysis"))
    parser.add_argument("--visual-count", type=int, default=6)
    args = parser.parse_args()

    input_dir = args.tp_dir / args.dataset
    with (input_dir / "true_positive_top_bits.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_tp = defaultdict(list)
    for row in rows:
        by_tp[int(row["tp_rank"])].append(row)
    if not by_tp or any(len(items) != 5 for items in by_tp.values()):
        raise ValueError("Every scaffold true positive must have exactly five feature rows")
    if any(float(row["shap_value"]) <= 0 for row in rows):
        raise ValueError("Found a non-positive SHAP value in the positive-only input")
    if any(sorted(int(row["bit_rank"]) for row in items) != [1, 2, 3, 4, 5] for items in by_tp.values()):
        raise ValueError("Top-five ranks are incomplete")

    output = args.output_dir / args.dataset
    visual_dir = output / "visual_check"
    visual_dir.mkdir(parents=True, exist_ok=True)
    detailed_rows = []
    compound_motif_shap = defaultdict(lambda: defaultdict(list))
    motif_fragments = defaultdict(Counter)
    visual_candidates = []

    for tp_rank, feature_rows in sorted(by_tp.items()):
        first = feature_rows[0]
        molecule = Chem.MolFromSmiles(first["smiles"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES for TP rank {tp_rank}")
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        merged_atoms, merged_bonds, motif_names = set(), set(), set()
        multiatom_features = 0
        for row in sorted(feature_rows, key=lambda value: int(value["bit_rank"])):
            bit = int(row["bit_index"])
            shap_value = float(row["shap_value"])
            if fingerprint[bit] <= 0 or bit not in bit_info:
                raise ValueError(f"TP {tp_rank}: selected bit {bit} is absent")
            feature_motifs = set()
            has_multiatom = False
            for occurrence, (center, radius) in enumerate(bit_info[bit], start=1):
                atoms, bonds = environment_atoms_and_bonds(molecule, center, radius)
                atom_tuple, bond_tuple = tuple(sorted(atoms)), tuple(sorted(bonds))
                fragment = fragment_smiles(molecule, atom_tuple, bond_tuple)
                motif = motif_class(molecule, atom_tuple, radius)
                feature_motifs.add(motif)
                motif_names.add(motif)
                motif_fragments[motif][fragment] += 1
                merged_atoms.update(atoms)
                merged_bonds.update(bonds)
                has_multiatom |= radius > 0
                detailed_rows.append(
                    {
                        "tp_rank": tp_rank,
                        "test_row_index": int(first["test_row_index"]),
                        "smiles": first["smiles"],
                        "predicted_probability": float(first["predicted_probability"]),
                        "feature_rank": int(row["bit_rank"]),
                        "bit_index": bit,
                        "positive_shap": shap_value,
                        "occurrence": occurrence,
                        "center_atom": center,
                        "radius": radius,
                        "atom_indices": ";".join(map(str, atom_tuple)),
                        "bond_indices": ";".join(map(str, bond_tuple)),
                        "fragment_smiles": fragment,
                        "motif_class": motif,
                    }
                )
            for motif in feature_motifs:
                compound_motif_shap[motif][tp_rank].append(shap_value)
            multiatom_features += int(has_multiatom)
        visual_candidates.append(
            (multiatom_features, len(motif_names), float(first["predicted_probability"]), tp_rank,
             int(first["test_row_index"]), first["smiles"], molecule, merged_atoms, merged_bonds, motif_names)
        )

    tp_count = len(by_tp)
    summary_rows = []
    for motif, compounds in compound_motif_shap.items():
        compound_values = [float(np.mean(values)) for values in compounds.values()]
        representative = motif_fragments[motif].most_common(1)[0][0]
        summary_rows.append(
            {
                "structural_motif": motif,
                "true_positive_compounds": len(compounds),
                "percent_all_tp": 100.0 * len(compounds) / tp_count,
                "mean_positive_shap": float(np.mean(compound_values)),
                "median_positive_shap": float(np.median(compound_values)),
                "representative_fragment": representative,
            }
        )
    summary_rows.sort(key=lambda row: (-row["true_positive_compounds"], row["structural_motif"]))

    selected_visuals = []
    covered_motifs = set()
    remaining = sorted(visual_candidates, reverse=True)
    while remaining and len(selected_visuals) < args.visual_count:
        best = max(
            remaining,
            key=lambda item: (
                len(item[-1] - covered_motifs), item[0], item[1], item[2]
            ),
        )
        remaining.remove(best)
        selected_visuals.append(best)
        covered_motifs.update(best[-1])

    manifest_rows = []
    for _, _, probability, tp_rank, row_index, smiles, molecule, atoms, bonds, motifs in selected_visuals:
        stem = visual_dir / f"tp_{tp_rank:04d}_testrow_{row_index}"
        draw_clean(molecule, atoms, bonds, stem)
        manifest_rows.append(
            {
                "tp_rank": tp_rank,
                "test_row_index": row_index,
                "smiles": smiles,
                "predicted_probability": probability,
                "motif_classes": ";".join(sorted(motifs)),
                "png": stem.with_suffix(".png").name,
                "svg": stem.with_suffix(".svg").name,
            }
        )

    for path, output_rows in (
        (output / "recurrent_structural_motifs.csv", summary_rows),
        (output / "all_tp_top5_positive_environments.csv", detailed_rows),
        (visual_dir / "manifest.csv", manifest_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    metadata = {
        "dataset": args.dataset,
        "scope": "true positives from held-out scaffold test only",
        "true_positive_count": tp_count,
        "features_per_compound": 5,
        "feature_selection": "top five positive local SHAP values among present ECFP bits",
        "global_mean_abs_shap_used": False,
        "ten_fold_used": False,
        "external_or_az_series_used": False,
        "prevalence_unit": "unique compound",
        "shap_aggregation": "mean per compound-motif, then mean/median across compounds",
        "visual_count": len(manifest_rows),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"true positives: {tp_count}")
    print(f"feature rows: {len(rows)}")
    print(f"environment occurrences: {len(detailed_rows)}")
    print(f"motif classes: {len(summary_rows)}")
    print(f"visual checks: {len(manifest_rows)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
