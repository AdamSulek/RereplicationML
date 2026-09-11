#!/usr/bin/env python3
"""Per-compound mapping from count-ECFP SHAP values to RDKit environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


FEATURE_LENGTH = 2048


@dataclass(frozen=True)
class Environment:
    center_atom: int
    radius: int
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]


@dataclass(frozen=True)
class LocalBit:
    rank: int
    bit_index: int
    shap_value: float
    count: int
    environments: tuple[Environment, ...]


def count_ecfp_and_bit_info(
    molecule: Chem.Mol,
    *,
    radius: int = 2,
    fp_size: int = FEATURE_LENGTH,
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, int], ...]]]:
    """Compute the model count fingerprint and its provenance in one RDKit call."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
        countSimulation=False,
    )
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    fingerprint = generator.GetCountFingerprintAsNumPy(
        molecule, additionalOutput=additional_output
    )
    bit_info = {
        int(bit): tuple((int(center), int(env_radius)) for center, env_radius in occurrences)
        for bit, occurrences in additional_output.GetBitInfoMap().items()
    }
    return np.asarray(fingerprint), bit_info


def environment_atoms_and_bonds(
    molecule: Chem.Mol, center: int, radius: int
) -> tuple[set[int], set[int]]:
    """Return the complete circular environment encoded by one Morgan occurrence."""
    bonds = set(Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center))
    atoms = {center}
    for bond_index in bonds:
        bond = molecule.GetBondWithIdx(bond_index)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return atoms, bonds


def verify_fingerprint_row(expected: Sequence[float], actual: Sequence[float]) -> None:
    """Fail loudly before drawing when the molecule is not the model input row."""
    expected_array = np.asarray(expected).reshape(-1)
    actual_array = np.asarray(actual).reshape(-1)
    if expected_array.shape != actual_array.shape:
        raise ValueError(
            f"Fingerprint shape mismatch: model row {expected_array.shape}, "
            f"recomputed {actual_array.shape}"
        )
    differing = np.flatnonzero(expected_array != actual_array)
    if differing.size:
        preview = ", ".join(map(str, differing[:10]))
        raise ValueError(
            f"Fingerprint mismatch at {differing.size} feature(s), first indices: {preview}"
        )


def select_local_bits(
    molecule: Chem.Mol,
    fingerprint: Sequence[float],
    bit_info: Mapping[int, Iterable[tuple[int, int]]],
    shap_values: Sequence[float],
    *,
    top_n: int,
    positive_only: bool = False,
) -> list[LocalBit]:
    """Select top bits solely from one molecule's fingerprint and SHAP vector."""
    fingerprint_array = np.asarray(fingerprint).reshape(-1)
    shap_array = np.asarray(shap_values).reshape(-1)
    if fingerprint_array.shape != shap_array.shape:
        raise ValueError(
            f"Fingerprint/SHAP shape mismatch: {fingerprint_array.shape} vs {shap_array.shape}"
        )
    if top_n < 1:
        raise ValueError("top_n must be positive")

    candidates = []
    for bit in np.flatnonzero(fingerprint_array):
        bit = int(bit)
        value = float(shap_array[bit])
        if bit not in bit_info or (positive_only and value <= 0):
            continue
        candidates.append((bit, value))
    candidates.sort(key=lambda item: (-abs(item[1]), item[0]))

    selected = []
    for rank, (bit, value) in enumerate(candidates[:top_n], start=1):
        environments = []
        for center, env_radius in bit_info[bit]:
            atoms, bonds = environment_atoms_and_bonds(molecule, center, env_radius)
            environments.append(Environment(int(center), int(env_radius), tuple(sorted(atoms)), tuple(sorted(bonds))))
        selected.append(LocalBit(rank, bit, value, int(fingerprint_array[bit]), tuple(environments)))
    return selected
