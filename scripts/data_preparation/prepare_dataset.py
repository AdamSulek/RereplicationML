#!/usr/bin/env python3
"""Reference preparation matching the manuscript's documented data procedure.

This implementation is reproducible, but is not claimed to be the historical
source of the released split assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold
from skfp.model_selection import scaffold_train_valid_test_split


SEED = 42


def canonical_largest_fragment(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError(f"SMILES produced no molecular fragments: {smiles!r}")
    ranked = sorted(
        fragments,
        key=lambda mol: (-mol.GetNumHeavyAtoms(), Chem.MolToSmiles(mol, canonical=True)),
    )
    return Chem.MolToSmiles(ranked[0], canonical=True, isomericSmiles=True)


def activity_labels(series: pd.Series, positive: set[str], negative: set[str]) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    overlap = positive & negative
    if overlap:
        raise ValueError(f"Labels cannot be both positive and negative: {sorted(overlap)}")
    unknown = sorted(set(normalized) - positive - negative)
    if unknown:
        raise ValueError(f"Unmapped activity label(s): {unknown[:20]}")
    return normalized.map(lambda value: 1 if value in positive else 0).astype(int)


def assign_reference_splits(frame: pd.DataFrame, *, seed: int = SEED) -> pd.Series:
    train_idx, valid_idx, test_idx = scaffold_train_valid_test_split(
        frame["smiles"].tolist(),
        train_size=0.8,
        valid_size=0.1,
        test_size=0.1,
        use_csk=False,
        return_indices=True,
    )
    assignments = pd.Series(index=frame.index, dtype="object")
    assignments.loc[list(valid_idx)] = "val"
    assignments.loc[list(test_idx)] = "test"
    outer_train = frame.loc[list(train_idx)]
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, fold_indices) in enumerate(
        splitter.split(outer_train["smiles"], outer_train["activity"])
    ):
        assignments.loc[outer_train.index[fold_indices]] = f"train_{fold}"
    if assignments.isna().any():
        raise RuntimeError("Some rows were not assigned to a split")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="activity")
    parser.add_argument("--positive-label", action="append", default=None)
    parser.add_argument("--negative-label", action="append", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {args.smiles_column, args.label_column}
    missing = required - set(source.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    prepared = pd.DataFrame(
        {
            "source_row_index": source.index.astype(int),
            "smiles": source[args.smiles_column].map(canonical_largest_fragment),
            "activity": activity_labels(
                source[args.label_column],
                {value.casefold() for value in (args.positive_label or ["1", "active"])},
                {value.casefold() for value in (args.negative_label or ["0", "inactive"])},
            ),
        }
    )
    conflicts = prepared.groupby("smiles")["activity"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting labels for {len(conflicts)} canonical SMILES")
    before_deduplication = len(prepared)
    prepared = prepared.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    prepared["split"] = assign_reference_splits(prepared, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)
    metadata = {
        "status": "reference implementation; historical split identity is not claimed",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "canonicalization": "RDKit canonical isomeric SMILES",
        "fragment_policy": "largest heavy-atom fragment; canonical-SMILES tie break",
        "duplicate_policy": "one row per canonical SMILES; conflicting labels are errors",
        "rows_before_deduplication": before_deduplication,
        "rows_after_deduplication": len(prepared),
        "scaffold_split": "scikit-fingerprints deterministic Bemis-Murcko core scaffold, 80/10/10, use_csk=False",
        "train_shards": "StratifiedKFold(10, shuffle=True) within scaffold training rows",
        "seed": args.seed,
        "split_counts": prepared["split"].value_counts().sort_index().to_dict(),
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
