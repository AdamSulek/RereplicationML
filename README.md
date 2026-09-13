# RereplicationML

Reproducible machine-learning workflows for prioritizing compounds associated with the DNA re-replication phenotype. The primary manuscript models are XGBoost and R-MAT. Random Forest and linear SVM are retained for the supplementary benchmark table.

## Data and checkpoints

Prepared datasets, model inputs, inference inputs, and final checkpoints are distributed through the [project Hugging Face dataset](https://huggingface.co/datasets/klimczakjakubdev/rereplication-ml).

| Release path | Contents |
| --- | --- |
| `raw/mcf10a_raw.parquet` | Curated MCF10A SMILES, binary `activity`, and exact scaffold/CV `split`. |
| `raw/sw480_raw.parquet` | Curated SW480-clean SMILES, binary `activity`, and exact scaffold/CV `split`. |
| `input_data/ecfp/mcf10a_ecfp.parquet` | MCF10A count Morgan/ECFP R2/2048 used by XGBoost. |
| `input_data/ecfp/sw480_ecfp.parquet` | SW480-clean count Morgan/ECFP R2/2048 used by XGBoost. |
| `input_data/rmat/{mcf10a,sw480}/` | R-MAT `train_0.p` ... `train_9.p`, `val.p`, and `test.p`. |
| `checkpoints/xgb/` | One scaffold and ten fold-specific saved XGBoost models per dataset. |
| `checkpoints/rmat/` | One scaffold and ten fold-specific R-MAT checkpoints per dataset. |
| `inference/` | Prepared AZ compound inputs for XGBoost and R-MAT inference. |

```bash
hf download klimczakjakubdev/rereplication-ml \
  --repo-type dataset \
  --local-dir ./downloaded_release
```

The commands below use `downloaded_release/` directly. To use the training defaults, create the expected local paths once:

```bash
mkdir -p data/mcf10a data/sw480 pickle_dataloaders
ln -sfn ../../downloaded_release/input_data/ecfp/mcf10a_ecfp.parquet \
  data/mcf10a/mcf10a_xgb_2048.parquet
ln -sfn ../../downloaded_release/input_data/ecfp/sw480_ecfp.parquet \
  data/sw480/sw480_clean_xgb.parquet
ln -sfn ../downloaded_release/input_data/rmat/mcf10a pickle_dataloaders/mcf10a
ln -sfn ../downloaded_release/input_data/rmat/sw480 pickle_dataloaders/sw480_clean
```

## Installation

```bash
conda env create -f environment.yml
conda activate rereplication-ml
```

`huggingface_hub` is installed by `environment.yml`. If `hf` is missing in an older environment, run `python -m pip install -U huggingface_hub`; authentication is required for uploads (`hf auth login`) but not for this public download.

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
mkdir -p outputs
python -c "import pandas as pd; pd.read_parquet('downloaded_release/raw/mcf10a_raw.parquet', columns=['smiles', 'activity']).to_csv('outputs/mcf10a_reference_input.csv', index=False)"
python scripts/data_preparation/prepare_dataset.py \
  --input outputs/mcf10a_reference_input.csv \
  --output outputs/prepared_reference.parquet
```

This reference implementation uses RDKit canonical isomeric SMILES, selects the largest heavy-atom fragment with a deterministic tie break, removes duplicate canonical SMILES, rejects conflicting duplicate labels, creates a deterministic 80/10/10 Bemis-Murcko core-scaffold split through scikit-fingerprints, and divides the scaffold-training subset into ten deterministic stratified shards. It is not claimed to be the historical source of the released assignments.

## Generate XGBoost ECFP features

```bash
python scripts/data_preparation/generate_ecfp.py \
  --input downloaded_release/raw/mcf10a_raw.parquet \
  --output outputs/mcf10a_ecfp_rebuilt.parquet
```

The representation is fully fixed: Morgan/ECFP radius 2, 2048 integer count features, chirality disabled, bond types enabled, ring membership enabled, and count simulation disabled. The feature column is `X_morgan_radius_2_count`.

## Generate R-MAT features

```bash
python scripts/data_preparation/featurize_rmat.py \
  --dataset mcf10a \
  --input downloaded_release/raw/mcf10a_raw.parquet \
  --output-dir outputs/rmat/mcf10a
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
  --selection-criterion val_loss
```

Weighted focal loss:

```bash
python scripts/train/train_rmat.py \
  --dataset mcf10a --split-mode scaffold \
  --loss focal --focal-gamma 2 --lr 2.5e-5 \
  --selection-criterion val_loss
```

`--selection-criterion` accepts `val_loss`, `val_roc_auc`, or `val_pr_auc`. The released checkpoints use weighted BCE and were selected by minimum `val_loss`; the other criteria are supported for controlled sensitivity runs. The chosen criterion controls both checkpoint replacement and early-stopping patience. Learning-rate/loss searches use repeated invocations of this same trainer; there is no separate hyperparameter-search implementation.

For fold 0 of the paper 10-fold protocol:

```bash
python scripts/train/train_rmat.py \
  --dataset mcf10a --split-mode kfold --fold 0 \
  --loss bce --lr 2.5e-5 \
  --selection-criterion val_loss
```

Repeat `--fold 0` through `--fold 9`. Each run records configuration, source pickle assignments, validation history, selected criterion/value/epoch, checkpoint metadata, and final test metrics.

## Evaluate saved models

```bash
python scripts/model_evaluation/evaluate_xgb.py \
  --dataset mcf10a --split test \
  --data-path downloaded_release/input_data/ecfp/mcf10a_ecfp.parquet \
  --model-path downloaded_release/checkpoints/xgb/mcf10a/scaffold/model.json \
  --predictions outputs/mcf10a_xgb_scaffold_test.csv

python scripts/model_evaluation/evaluate_rmat.py \
  --checkpoint downloaded_release/checkpoints/rmat/mcf10a/scaffold.pt \
  --split-pickle downloaded_release/input_data/rmat/mcf10a/test.p \
  --predictions outputs/mcf10a_rmat_scaffold_test.csv
```

Reported screening metrics are ROC-AUC, average precision (called PR-AUC), F1, recall, precision, active prevalence, PR-AUC divided by prevalence, EF@1%, and EF@5%. Threshold metrics default to 0.5. Ranking cutoffs use `ceil(N × fraction)` after a stable descending score sort.

Model selection always uses validation data. Final test metrics must not be used to choose the loss, learning rate, or checkpoint criterion.

## Predict external SMILES

The input must be a CSV containing `smiles`. The released AZ input is `downloaded_release/inference/az_compounds_ecfp.csv`; [examples/example_smiles.csv](examples/example_smiles.csv) is a minimal smoke-test input.

```bash
python scripts/model_evaluation/predict_xgb.py \
  --input downloaded_release/inference/az_compounds_ecfp.csv \
  --model-path downloaded_release/checkpoints/xgb/mcf10a/scaffold/model.json \
  --output outputs/az_predictions_xgb_mcf10a.csv

python scripts/model_evaluation/predict_rmat.py \
  --input downloaded_release/inference/az_compounds_ecfp.csv \
  --checkpoint downloaded_release/checkpoints/rmat/mcf10a/scaffold.pt \
  --output outputs/az_predictions_rmat_mcf10a.csv
```

Both commands load saved models and return a `prediction_score`; they do not retrain models.

## Applicability domain

```bash
python scripts/model_evaluation/applicability_domain.py \
  --input downloaded_release/inference/az_compounds_ecfp.csv \
  --reference downloaded_release/raw/mcf10a_raw.parquet \
  --output outputs/az_applicability_domain_mcf10a.csv
```

AD uses a binary Morgan radius-2, 2048-bit fingerprint, Tanimoto similarity, the arithmetic mean of the five nearest training-reference similarities, and the fixed cutoff `0.49938`. This binary similarity fingerprint is deliberately distinct from the count ECFP used by XGBoost.

## XGBoost SHAP interpretation

The audited manuscript workflow is restricted to held-out scaffold true positives and local positive SHAP features present in each compound:

```bash
python scripts/interpretability/analyze_xgb_shap.py \
  --dataset mcf10a \
  --model-path downloaded_release/checkpoints/xgb/mcf10a/scaffold/model.json \
  --external downloaded_release/inference/az_compounds_ecfp.csv \
  --save-test-values
python scripts/interpretability/draw_xgb_tp_shap.py \
  --dataset mcf10a \
  --model-path downloaded_release/checkpoints/xgb/mcf10a/scaffold/model.json
python scripts/interpretability/summarize_xgb_tp_motifs.py --dataset mcf10a
python scripts/interpretability/export_mcf10a_tp_ring_analysis.py
```

The scripts load the exact saved scaffold model; they do not retrain it. The mapping code computes TreeSHAP separately for every molecule, recomputes the exact count ECFP, verifies it against the model input row, restricts selection to nonzero bits present in that molecule, maps all hashed-feature occurrences to atoms and bonds, and excludes absent bits. It does not map a global mean-absolute-SHAP ranking back onto individual molecules. External compound examples can be drawn with `draw_xgb_shap_examples.py`. The older binary-bit/global-ranking SHAP mapping is not part of the public workflow.

## Visualization and supplementary models

Plotting-only utilities are under `scripts/visualization/`. For the fixed binary Morgan R2/2048 chemical-space projection:

```bash
python scripts/visualization/plot_umap.py \
  --input downloaded_release/input_data/ecfp/mcf10a_ecfp.parquet \
  --output outputs/mcf10a_umap.png \
  --seed 42
```

Random Forest and linear SVM training/evaluation scripts are under `scripts/supplementary_models/`; their scientific implementations have not been redesigned.

## Reproducibility smoke checks

After installation and release download, run:

```bash
python -m unittest discover -s tests
python scripts/model_evaluation/predict_xgb.py \
  --input examples/example_smiles.csv \
  --model-path downloaded_release/checkpoints/xgb/mcf10a/scaffold/model.json \
  --output outputs/example_xgb_predictions.csv
python scripts/model_evaluation/predict_rmat.py \
  --input examples/example_smiles.csv \
  --checkpoint downloaded_release/checkpoints/rmat/mcf10a/scaffold.pt \
  --output outputs/example_rmat_predictions.csv
```
