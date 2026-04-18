# geo-rf-xgboost-catboost

[English](./README.md) | [简体中文](./README.zh-CN.md)

Tree-based geochemical regression workflow with optional spatial partitioning.

## What changed

The repository now supports the bundled `Sn.xlsx` and `Ta.xlsx` files as a first-class example workflow:

- no hard-coded absolute paths
- one shared pipeline module for data loading, training, prediction, and metrics
- optional Geo models only when coordinate columns are available
- CSV-based outputs for example data, with shapefile prediction export kept as an optional path when `geopandas` is installed
- plotting scripts read generated metrics/predictions instead of hard-coded values

## Quick start

1. Install the core dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare the bundled example dataset:

```bash
python 0_dataprepare.py
```

3. Train the available models:

```bash
python 1_train_geo_rf_xgb_catboost.py
```

If `xgboost` or `catboost` is not installed, the script will train the available subset and report what was skipped.

4. Generate predictions for the example data:

```bash
python 3_generate_prediction_of_train.py
```

5. Plot evaluation figures:

```bash
python 4_plot_scatter.py --target Sn
python 4_plot_scatter_all_in_one.py --target Ta
python 5_plot_radar.py
```

## Input formats

The pipeline supports:

- `.xlsx` and `.csv` for tabular workflows
- `.shp` for spatial workflows when `geopandas` is installed

For bundled examples, `0_dataprepare.py` merges `Sn.xlsx` and `Ta.xlsx` into `data/input/example_dataset.csv`.

## Outputs

Generated artifacts are written under `results/`:

- `results/models/`: trained model files
- `results/metrics/model_metrics.csv`: evaluation summary
- `results/metadata/training_metadata.json`: feature/target/model metadata for prediction reuse
- `results/predictions/`: holdout predictions and full-dataset predictions
- `results/feature_importance/`: feature importance CSV and PNG files

Figures are written under `figures/`.
