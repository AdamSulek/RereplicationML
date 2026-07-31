#!/usr/bin/env python3
"""Add a requested molecular fingerprint column to a parquet dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from skfp.fingerprints import ECFPFingerprint, KlekotaRothFingerprint, MordredFingerprint


LOGGER = logging.getLogger(__name__)


def build_fingerprint(name: str):
    if name == "ecfp_2":
        return ECFPFingerprint(count=False, radius=2)
    if name == "ecfp_count_2":
        return ECFPFingerprint(count=True, radius=2)
    if name == "ecfp_3":
        return ECFPFingerprint(count=False, radius=3)
    if name == "ecfp_count_3":
        return ECFPFingerprint(count=True, radius=3)
    if name == "klekota":
        return KlekotaRothFingerprint(count=False)
    if name == "klekota_count":
        return KlekotaRothFingerprint(count=True)
    if name == "mordred":
        return MordredFingerprint()
    raise ValueError(f"Unknown fingerprint: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument(
        "--fingerprint",
        choices=["ecfp_2", "ecfp_count_2", "ecfp_3", "ecfp_count_3", "klekota", "klekota_count", "mordred"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    input_path = args.data_dir / args.dataset / "raw.parquet"
    frame = pd.read_parquet(input_path)
    if "smiles" not in frame.columns:
        raise KeyError("The dataset must contain a smiles column.")

    column = f"X_{args.fingerprint}"
    LOGGER.info("Generating %s for %d molecules", column, len(frame))
    frame[column] = list(build_fingerprint(args.fingerprint).transform(frame["smiles"].tolist()))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_path, index=False)
    LOGGER.info("Wrote %s", args.output_path)


if __name__ == "__main__":
    main()
