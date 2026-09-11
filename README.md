# RereplicationML

Reproducible machine-learning workflows for prioritizing compounds associated with the DNA re-replication phenotype. The primary manuscript models are XGBoost and R-MAT. Random Forest and linear SVM are retained for the supplementary benchmark table.

## Data and checkpoints

Prepared datasets and checkpoints are distributed separately through the [project Hugging Face dataset](https://huggingface.co/datasets/klimczakjakubdev/rereplication-ml). The repository's `data/` layout is intentionally unchanged during this code revision.

```bash
hf download klimczakjakubdev/rereplication-ml \
  --repo-type dataset \
  --local-dir ./downloaded_release
```

Consult the release manifest before placing downloaded files in their documented paths. Publication of the final Hugging Face layout and checksums is handled separately.

## Installation

```bash
conda env create -f environment.yml
conda activate rereplication-ml
```

R-MAT training normally requires a CUDA-compatible PyTorch build. Install the build appropriate for the reviewer machine or cluster before training R-MAT. The other workflows can run on CPU.

## Data protocols

### Held-out scaffold protocol

- `train_0` through `train_9` are combined for fitting.
- `val` is used only for model/checkpoint selection.
- `test` remains held out until selection is final and is then evaluated once.

### Paper 10-fold protocol

For fold `i`, the test set is `train_i`, validation is `train_((i + 1) mod 10)`, and training uses the other eight `train_*` shards plus the original `val` and `test` shards.

This reproduces the established R-MAT-compatible protocol. It is not an independent evaluation of the original scaffold test partition.

## Optional reference preprocessing

Use released assignments for exact manuscript reproduction. To create a new reference preparation:

```bash
python scripts/data_preparation/prepare_dataset.py \
  --input raw_compounds.csv \
  --output prepared_reference.parquet
```

This reference implementation uses RDKit canonical isomeric SMILES, selects the largest heavy-atom fragment with a deterministic tie break, removes duplicate canonical SMILES, rejects conflicting duplicate labels, creates a deterministic 80/10/10 Bemis-Murcko core-scaffold split through scikit-fingerprints, and divides the scaffold-training subset into ten deterministic stratified shards. It is not claimed to be the historical source of the released assignments.

## Generate XGBoost ECFP features

```bash
python scripts/data_preparation/generate_ecfp.py \
  --input prepared.parquet \
  --output prepared_ecfp.parquet
```

The representation is fully fixed: Morgan/ECFP radius 2, 2048 integer count features, chirality disabled, bond types enabled, ring membership enabled, and count simulation disabled. The feature column is `X_morgan_radius_2_count`.

## Generate R-MAT features

```bash
python scripts/data_preparation/featurize_rmat.py \
  --dataset mcf10a \
  --input prepared.parquet \
  --output-dir pickle_dataloaders/mcf10a
```

By default this writes `train_0.p` through `train_9.p`, `val.p`, and `test.p` with `RMatFeaturizer.from_pretrained("rmat_4M")`.

## Train XGBoost

```bash
python scripts/train/train_xgb.py \
  --dataset mcf10a \
  --mode scaffold \
  --selection-metric roc_auc

python scripts/train/train_xgb.py --dataset mcf10a --mode kfold
```

Scaffold hyperparameters are chosen on validation data. The selected scaffold model is evaluated once on the held-out scaffold test data. The 10-fold mode uses frozen selected hyperparameters and follows the paper protocol above.

## Train R-MAT

Weighted BCE:

```bash
python scripts/train/train_rmat.py \
  --dataset mcf10a --split-mode scaffold \
  --loss bce --lr 2.5e-5 \
  --selection-criterion val_pr_auc
```

Weighted focal loss:

```bash
python scripts/train/train_rmat.py \
  --dataset mcf10a --split-mode scaffold \
  --loss focal --focal-gamma 2 --lr 2.5e-5 \
  --selection-criterion val_pr_auc
```

`--selection-criterion` accepts `val_loss`, `val_roc_auc`, or `val_pr_auc`. The chosen criterion controls both checkpoint replacement and early-stopping patience. Learning-rate/loss searches use repeated invocations of this same trainer; there is no separate hyperparameter-search implementation.

For fold 0 of the paper 10-fold protocol:

```bash
python scripts/train/train_rmat.py \
  --dataset mcf10a --split-mode kfold --fold 0 \
  --loss focal --lr 2.5e-5 \
  --selection-criterion val_pr_auc
```

Repeat `--fold 0` through `--fold 9`. Each run records configuration, source pickle assignments, validation history, selected criterion/value/epoch, checkpoint metadata, and final test metrics.

## Evaluate saved models

```bash
python scripts/model_evaluation/evaluate_xgb.py \
  --dataset mcf10a --split test \
  --model-path artifacts/xgb_full/mcf10a/scaffold/model.json

python scripts/model_evaluation/evaluate_rmat.py \
  --checkpoint path/to/best_pr_auc.pt \
  --split-pickle pickle_dataloaders/mcf10a/test.p
```

Reported screening metrics are ROC-AUC, average precision (called PR-AUC), F1, recall, precision, active prevalence, PR-AUC divided by prevalence, EF@1%, and EF@5%. Threshold metrics default to 0.5. Ranking cutoffs use `ceil(N × fraction)` after a stable descending score sort.

Model selection always uses validation data. Final test metrics must not be used to choose the loss, learning rate, or checkpoint criterion.

## Predict external SMILES

The input must be a CSV containing `smiles`; [examples/example_smiles.csv](examples/example_smiles.csv) is a minimal example.

```bash
python scripts/model_evaluation/predict_xgb.py \
  --input examples/example_smiles.csv \
  --model-path path/to/model.json \
  --output predictions_xgb.csv

python scripts/model_evaluation/predict_rmat.py \
  --input examples/example_smiles.csv \
  --checkpoint path/to/checkpoint.pt \
  --output predictions_rmat.csv
```

Both commands load saved models and return a `prediction_score`; they do not retrain models.

## Applicability domain

```bash
python scripts/model_evaluation/applicability_domain.py \
  --input examples/example_smiles.csv \
  --reference data/mcf10a/raw.parquet \
  --output applicability_domain.csv
```

AD uses a binary Morgan radius-2, 2048-bit fingerprint, Tanimoto similarity, the arithmetic mean of the five nearest training-reference similarities, and the fixed cutoff `0.49938`. This binary similarity fingerprint is deliberately distinct from the count ECFP used by XGBoost.

## XGBoost SHAP interpretation

The audited manuscript workflow is restricted to held-out scaffold true positives and local positive SHAP features present in each compound:

```bash
python scripts/interpretability/draw_xgb_tp_shap.py --dataset mcf10a
python scripts/interpretability/summarize_xgb_tp_motifs.py --dataset mcf10a
python scripts/interpretability/export_mcf10a_tp_ring_analysis.py
```

The mapping code recomputes the exact count ECFP, verifies it against the model row, maps all hashed-feature occurrences to atoms and bonds, and excludes absent bits. External compound examples can be drawn with `draw_xgb_shap_examples.py`. The older binary-bit SHAP mapping is not part of the public workflow.

## Visualization and supplementary models

Plotting-only utilities are under `scripts/visualization/`, including UMAP, saved performance tables, and SHAP plots. Random Forest and linear SVM training/evaluation scripts are under `scripts/supplementary_models/`; their scientific implementations have not been redesigned.
