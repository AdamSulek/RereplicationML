#!/usr/bin/env python3
"""Export clean TP drawings and ring classes from saved local SHAP features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.interpretability.shap_structure_mapping import count_ecfp_and_bit_info, environment_atoms_and_bonds


RING_CLASSES = (
    "N-containing 6-membered aromatic ring",
    "N-containing 5-membered aromatic ring",
    "fused N-heteroaromatic system",
    "O-containing heteroaromatic ring",
    "S-containing heteroaromatic ring",
    "multiple-N heteroaromatic ring",
    "carbocyclic aromatic ring",
    "fused carbocyclic aromatic system",
)


def fragment_smiles(molecule, atoms, bonds):
    return Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=sorted(atoms),
        bondsToUse=sorted(bonds),
        isomericSmiles=True,
    )


def aromatic_rings(molecule):
    return [
        tuple(map(int, ring))
        for ring in molecule.GetRingInfo().AtomRings()
        if len(ring) in {5, 6}
        and all(molecule.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring)
    ]


def fused_components(rings):
    components = []
    unseen = set(range(len(rings)))
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            current = stack.pop()
            neighbours = {
                other
                for other in list(unseen)
                if len(set(rings[current]) & set(rings[other])) >= 2
            }
            unseen -= neighbours
            component |= neighbours
            stack.extend(neighbours)
        components.append(component)
    return components


def cyclic_distance(size, first, second):
    distance = abs(first - second)
    return min(distance, size - distance)


def ring_subcategory(molecule, ring):
    symbols = [molecule.GetAtomWithIdx(atom).GetSymbol() for atom in ring]
    n_positions = [index for index, symbol in enumerate(symbols) if symbol == "N"]
    o_positions = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    s_positions = [index for index, symbol in enumerate(symbols) if symbol == "S"]
    size = len(ring)
    if size == 6:
        if len(n_positions) == 1 and not o_positions and not s_positions:
            return "pyridine-like"
        if len(n_positions) == 2 and not o_positions and not s_positions:
            distance = cyclic_distance(size, *n_positions)
            return {1: "pyridazine-like", 2: "pyrimidine-like", 3: "pyrazine-like"}[distance]
        if len(n_positions) == 3 and not o_positions and not s_positions:
            return "triazine-like"
        if len(n_positions) == 4 and not o_positions and not s_positions:
            return "tetrazine-like"
    if size == 5:
        if len(n_positions) == 1 and not o_positions and not s_positions:
            nitrogen = molecule.GetAtomWithIdx(ring[n_positions[0]])
            if nitrogen.GetTotalNumHs() > 0:
                return "pyrrole-like"
        if not n_positions and len(o_positions) == 1 and not s_positions:
            return "furan-like"
        if not n_positions and not o_positions and len(s_positions) == 1:
            return "thiophene-like"
        if len(n_positions) == 2 and not o_positions and not s_positions:
            return "pyrazole-like" if cyclic_distance(size, *n_positions) == 1 else "imidazole-like"
        if len(n_positions) == 3 and not o_positions and not s_positions:
            return "triazole-like"
        if len(n_positions) == 1 and len(o_positions) == 1 and not s_positions:
            distance = cyclic_distance(size, n_positions[0], o_positions[0])
            return "isoxazole-like" if distance == 1 else "oxazole-like"
        if len(n_positions) == 1 and not o_positions and len(s_positions) == 1:
            distance = cyclic_distance(size, n_positions[0], s_positions[0])
            return "isothiazole-like" if distance == 1 else "thiazole-like"
    return ""


def ring_annotations(molecule, center, radius, environment_atoms):
    if radius == 0:
        return [], [], []
    rings = aromatic_rings(molecule)
    components = fused_components(rings)
    associated = [index for index, ring in enumerate(rings) if center in ring]
    classes, subcategories, ring_records = set(), set(), []
    for ring_index in associated:
        ring = rings[ring_index]
        atoms = [molecule.GetAtomWithIdx(atom) for atom in ring]
        symbols = [atom.GetSymbol() for atom in atoms]
        n_count = symbols.count("N")
        if len(ring) == 6 and n_count:
            classes.add(RING_CLASSES[0])
        if len(ring) == 5 and n_count:
            classes.add(RING_CLASSES[1])
        if "O" in symbols:
            classes.add(RING_CLASSES[3])
        if "S" in symbols:
            classes.add(RING_CLASSES[4])
        if n_count >= 2:
            classes.add(RING_CLASSES[5])
        if all(symbol == "C" for symbol in symbols):
            classes.add(RING_CLASSES[6])
        component = next(group for group in components if ring_index in group)
        component_atoms = set().union(*(set(rings[index]) for index in component))
        if len(component) > 1:
            if any(molecule.GetAtomWithIdx(atom).GetSymbol() == "N" for atom in component_atoms):
                classes.add(RING_CLASSES[2])
            if all(molecule.GetAtomWithIdx(atom).GetSymbol() == "C" for atom in component_atoms):
                classes.add(RING_CLASSES[7])
        subtype = ring_subcategory(molecule, ring)
        if subtype:
            subcategories.add(subtype)
        ring_records.append(
            {
                "ring_atoms": ";".join(map(str, sorted(ring))),
                "ring_size": len(ring),
                "ring_elements": "".join(symbols),
                "full_ring_in_environment": set(ring) <= set(environment_atoms),
                "subcategory": subtype,
            }
        )
    return sorted(classes), sorted(subcategories), ring_records


def draw_positive_environments(molecule, atoms, bonds, output):
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 720)
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().continuousHighlight = True
    atom_colors = {atom: (1.0, 0.55, 0.10) for atom in atoms}
    bond_colors = {bond: (1.0, 0.30, 0.05) for bond in bonds}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=sorted(atoms),
        highlightAtomColors=atom_colors,
        highlightBonds=sorted(bonds),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    output.write_bytes(drawer.GetDrawingText())


def create_contact_sheets(manifest, png_dir, output_dir, per_page=50):
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    columns, rows_per_page = 5, 10
    tile_width, image_height, caption_height = 300, 216, 44
    tile_height = image_height + caption_height
    for page_index in range(math.ceil(len(manifest) / per_page)):
        page_rows = manifest[page_index * per_page : (page_index + 1) * per_page]
        sheet = Image.new("RGB", (columns * tile_width, rows_per_page * tile_height), "white")
        canvas = ImageDraw.Draw(sheet)
        for offset, row in enumerate(page_rows):
            image = Image.open(png_dir / Path(row["png_path"]).name).convert("RGB")
            image.thumbnail((tile_width, image_height))
            x = (offset % columns) * tile_width + (tile_width - image.width) // 2
            y = (offset // columns) * tile_height + (image_height - image.height) // 2
            sheet.paste(image, (x, y))
            caption = f"{row['compound_id']}  p={float(row['predicted_probability']):.3f}"
            canvas.text(((offset % columns) * tile_width + 8, (offset // columns) * tile_height + image_height + 8), caption, fill="black", font=font)
        sheet.save(output_dir / f"contact_sheet_{page_index + 1:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/xgb_full/tp_shap/mcf10a/true_positive_top_bits.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/xgb_full/mcf10a_scaffold_tp_positive"))
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_tp = defaultdict(list)
    for row in source_rows:
        by_tp[int(row["tp_rank"])].append(row)
    if len(by_tp) != 1012:
        raise ValueError(f"Expected 1012 true positives, got {len(by_tp)}")
    if any(len(rows) != 5 for rows in by_tp.values()):
        raise ValueError("Every TP must have exactly five saved feature rows")
    if any(float(row["shap_value"]) <= 0 for row in source_rows):
        raise ValueError("Input contains a non-positive SHAP value")

    png_dir = args.output / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    audit_rows, manifest = [], []
    class_compound_bits = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    subclass_compound_bits = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    class_fragments = defaultdict(Counter)
    class_compounds = defaultdict(dict)
    subclass_fragments = defaultdict(Counter)

    for tp_rank, rows in sorted(by_tp.items()):
        rows.sort(key=lambda row: int(row["bit_rank"]))
        first = rows[0]
        molecule = Chem.MolFromSmiles(first["smiles"])
        fingerprint, bit_info = count_ecfp_and_bit_info(molecule)
        compound_id = f"MCF10A-TP-{tp_rank:04d}"
        merged_atoms, merged_bonds = set(), set()
        for row in rows:
            bit = int(row["bit_index"])
            shap_value = float(row["shap_value"])
            if bit not in bit_info or fingerprint[bit] <= 0:
                raise ValueError(f"{compound_id}: absent selected bit {bit}")
            feature_classes, feature_subclasses = set(), set()
            for occurrence, (center, radius) in enumerate(bit_info[bit], start=1):
                atoms, bonds = environment_atoms_and_bonds(molecule, center, radius)
                fragment = fragment_smiles(molecule, atoms, bonds)
                classes, subclasses, ring_records = ring_annotations(molecule, center, radius, atoms)
                feature_classes.update(classes)
                feature_subclasses.update(subclasses)
                for ring_class in classes:
                    class_fragments[ring_class][fragment] += 1
                for subclass in subclasses:
                    subclass_fragments[subclass][fragment] += 1
                merged_atoms.update(atoms)
                merged_bonds.update(bonds)
                audit_rows.append(
                    {
                        "compound_id": compound_id,
                        "tp_rank": tp_rank,
                        "test_row_id": int(first["test_row_index"]),
                        "predicted_probability": float(first["predicted_probability"]),
                        "smiles": first["smiles"],
                        "feature_rank": int(row["bit_rank"]),
                        "bit_index": bit,
                        "positive_shap": shap_value,
                        "occurrence": occurrence,
                        "center_atom": center,
                        "radius": radius,
                        "atom_indices": ";".join(map(str, sorted(atoms))),
                        "bond_indices": ";".join(map(str, sorted(bonds))),
                        "fragment_smiles": fragment,
                        "ring_classes": ";".join(classes),
                        "ring_subcategories": ";".join(subclasses),
                        "ring_details_json": json.dumps(ring_records, separators=(",", ":")),
                        "excluded_from_ring_ranking_as_r0": radius == 0,
                    }
                )
            for ring_class in feature_classes:
                class_compound_bits[ring_class][compound_id][bit].append(shap_value)
                class_compounds[ring_class][compound_id] = (tp_rank, first["smiles"], float(first["predicted_probability"]))
            for subclass in feature_subclasses:
                subclass_compound_bits[subclass][compound_id][bit].append(shap_value)

        png_name = f"{compound_id}_testrow_{int(first['test_row_index'])}.png"
        draw_positive_environments(molecule, merged_atoms, merged_bonds, png_dir / png_name)
        manifest.append(
            {
                "compound_id": compound_id,
                "tp_rank": tp_rank,
                "test_row_id": int(first["test_row_index"]),
                "predicted_probability": float(first["predicted_probability"]),
                "smiles": first["smiles"],
                "png_path": str((png_dir / png_name).resolve()),
            }
        )

    def summarize(mapping, fragments, include_representatives):
        summaries = []
        for name, compounds in mapping.items():
            compound_values = []
            for bits in compounds.values():
                unique_bit_values = [values[0] for values in bits.values()]
                compound_values.append(float(np.mean(unique_bit_values)))
            representative_compounds = ""
            if include_representatives:
                ranked = sorted(
                    compounds,
                    key=lambda compound: -float(np.mean([v[0] for v in compounds[compound].values()])),
                )[:5]
                representative_compounds = ";".join(ranked)
            summaries.append(
                {
                    "ring_class": name,
                    "n_true_positives": len(compounds),
                    "percent_of_1012_tp": 100.0 * len(compounds) / 1012,
                    "mean_positive_shap": float(np.mean(compound_values)) if compound_values else 0.0,
                    "median_positive_shap": float(np.median(compound_values)) if compound_values else 0.0,
                    "representative_environment": fragments[name].most_common(1)[0][0] if fragments[name] else "",
                    "representative_tp_compounds": representative_compounds,
                }
            )
        summaries.sort(key=lambda row: (-row["n_true_positives"], row["ring_class"]))
        return summaries

    ring_summary = summarize(class_compound_bits, class_fragments, True)
    missing = set(RING_CLASSES) - {row["ring_class"] for row in ring_summary}
    for name in missing:
        ring_summary.append(
            {"ring_class": name, "n_true_positives": 0, "percent_of_1012_tp": 0.0,
             "mean_positive_shap": 0.0, "median_positive_shap": 0.0,
             "representative_environment": "", "representative_tp_compounds": ""}
        )
    ring_summary.sort(key=lambda row: (-row["n_true_positives"], row["ring_class"]))
    subclass_summary = summarize(subclass_compound_bits, subclass_fragments, False)

    outputs = (
        (args.output / "manifest.csv", manifest),
        (args.output / "ring_environment_audit.csv", audit_rows),
        (args.output / "ring_class_summary.csv", ring_summary),
        (args.output / "ring_subcategory_summary.csv", subclass_summary),
    )
    for path, rows in outputs:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    create_contact_sheets(manifest, png_dir, args.output / "contact_sheets")
    metadata = {
        "scope": "1012 MCF10A XGBoost held-out scaffold true positives",
        "source": str(args.input.resolve()),
        "models_or_inference_run": False,
        "features_per_compound": 5,
        "all_feature_shap_values_positive": True,
        "global_mean_abs_shap_used": False,
        "prevalence_unit": "compound",
        "bit_occurrences_do_not_duplicate_shap": True,
        "ring_assignment": "aromatic ring containing the ECFP occurrence center; R0 excluded",
        "shap_aggregation": "unique bit within compound-class, mean within compound, then mean/median across compounds",
        "png_count": len(manifest),
        "contact_sheet_count": math.ceil(len(manifest) / 50),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
