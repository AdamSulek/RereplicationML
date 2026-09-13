#!/usr/bin/env python3
"""Apply the manuscript's fixed Morgan/Tanimoto applicability-domain rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


RADIUS = 2
FP_SIZE = 2048
TOP_K = 5
FIXED_CUTOFF = 0.49938


def make_bit_generator():
    """The AD bit fingerprint; deliberately distinct from XGBoost count ECFP."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
        countSimulation=False,
    )


def bit_fingerprints(smiles: list[str]):
    generator = make_bit_generator()
    fingerprints = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {index}: {value!r}")

        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def applicability_scores(query_smiles: list[str], reference_smiles: list[str]) -> list[dict[str, object]]:
    if len(reference_smiles) < TOP_K:
        raise ValueError(f"At least {TOP_K} reference molecules are required")
    reference = bit_fingerprints(reference_smiles)
    queries = bit_fingerprints(query_smiles)
    rows = []
    for query in queries:
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, reference), dtype=float)
        top = np.partition(similarities, -TOP_K)[-TOP_K:]
        mean_top5 = float(top.mean())
        rows.append(
            {
                "mean_top5_similarity": mean_top5,
                "nearest_neighbor_similarity": float(top.max()),
                "cutoff": FIXED_CUTOFF,
                "AD_status": "In-domain" if mean_top5 >= FIXED_CUTOFF else "Out-of-domain",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Query CSV containing smiles")
    parser.add_argument("--reference", type=Path, required=True, help="Prepared reference parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compound-column", default="compound")
    args = parser.parse_args()

    queries = pd.read_csv(args.input)
    if "smiles" not in queries:
        raise KeyError("Query CSV must contain a smiles column")
    reference = pd.read_parquet(args.reference, columns=["smiles", "split"])
    reference = reference[reference["split"].astype(str).str.startswith("train_")]
    if reference.empty:
        raise ValueError("Reference contains no train_* rows")

    scores = applicability_scores(
        queries["smiles"].astype(str).tolist(),
        reference["smiles"].astype(str).tolist(),
    )
    output = queries.copy()
    if args.compound_column not in output:
        output.insert(0, args.compound_column, [f"query_{index + 1}" for index in range(len(output))])
    for column in ("mean_top5_similarity", "nearest_neighbor_similarity", "cutoff", "AD_status"):
        output[column] = [row[column] for row in scores]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata = {
        "reference": str(args.reference.resolve()),
        "reference_filter": "split starts with train_",
        "reference_rows": int(len(reference)),
        "fingerprint": {
            "representation": "binary Morgan bit vector (not XGBoost count ECFP)",
            "radius": RADIUS,
            "fp_size": FP_SIZE,
            "include_chirality": False,
            "use_bond_types": True,
            "include_ring_membership": True,
            "count_simulation": False,
        },
        "similarity": "Tanimoto",
        "neighbors": TOP_K,
        "score": "arithmetic mean of five highest reference similarities",
        "cutoff": FIXED_CUTOFF,
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
