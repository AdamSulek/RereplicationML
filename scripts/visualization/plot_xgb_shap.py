#!/usr/bin/env python3
"""Create publication-friendly plots from saved native XGBoost SHAP outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], required=True)
    parser.add_argument("--shap-dir", type=Path, default=Path("artifacts/xgb_full/shap"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.shap_dir / args.dataset
    importance = pd.read_csv(output / "scaffold_test_feature_importance.csv")
    top = importance.head(20).sort_values("mean_abs_shap", ascending=True)

    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["feature_index"].astype(str), top["mean_abs_shap"], color="#176b87")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("ECFP feature index")
    axis.set_title(f"{args.dataset}: global SHAP importance")
    fig.tight_layout()
    fig.savefig(output / "shap_mean_abs_top20.png", dpi=220)
    plt.close(fig)

    signed = importance.reindex(
        importance["mean_shap"].abs().sort_values(ascending=False).index[:20]
    ).sort_values("mean_shap", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 7))
    colors = ["#b23a48" if value < 0 else "#176b87" for value in signed["mean_shap"]]
    axis.barh(signed["feature_index"].astype(str), signed["mean_shap"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean SHAP value")
    axis.set_ylabel("ECFP feature index")
    axis.set_title(f"{args.dataset}: signed mean SHAP contribution")
    fig.tight_layout()
    fig.savefig(output / "shap_signed_mean_top20.png", dpi=220)
    plt.close(fig)

    external = pd.read_csv(output / "external_shap_base.csv")
    values = np.load(output / "external_shap_values.npz")["values"]
    indices = importance.head(20)["feature_index"].to_numpy(dtype=int)
    fig_height = max(8, len(external) * 0.12)
    fig, axis = plt.subplots(figsize=(10, fig_height))
    image = axis.imshow(values[:, indices], aspect="auto", cmap="coolwarm", interpolation="nearest")
    axis.set_xlabel("Top global ECFP feature index")
    axis.set_ylabel("External compound")
    axis.set_xticks(np.arange(len(indices)))
    axis.set_xticklabels(indices, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(external)))
    axis.set_yticklabels(external.get("compound_id", pd.Series(range(len(external)))))
    axis.set_title(f"{args.dataset}: external SHAP contributions")
    fig.colorbar(image, ax=axis, label="SHAP value")
    fig.tight_layout()
    fig.savefig(output / "external_shap_heatmap_top20.png", dpi=180)
    plt.close(fig)

    print(f"wrote plots to: {output}")


if __name__ == "__main__":
    main()
