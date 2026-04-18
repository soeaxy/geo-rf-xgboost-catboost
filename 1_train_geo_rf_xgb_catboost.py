from __future__ import annotations

import argparse
from pathlib import Path

from geo_pipeline import (
    DEFAULT_ID_COLUMN,
    DEFAULT_MODEL_ORDER,
    DEFAULT_PREPARED_DATASET,
    DEFAULT_RESULTS_ROOT,
    DatasetError,
    ensure_example_dataset,
    train_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train RF/XGBoost/CatBoost models on the prepared dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PREPARED_DATASET,
        help="Input dataset (.csv, .xlsx, or .shp). Defaults to the prepared example dataset.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory used for saved models, metrics, predictions, and metadata.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target columns to train. Defaults to bundled targets detected in the dataset.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODEL_ORDER),
        help="Base model types to train. Supported values are rf, xgb, and cat.",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help="Sample id column.",
    )
    parser.add_argument(
        "--coord-columns",
        nargs=2,
        default=None,
        metavar=("X_COL", "Y_COL"),
        help="Optional coordinate columns. Geo models are trained only when coordinates are available.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout split ratio.",
    )
    parser.add_argument(
        "--max-geo-depth",
        type=int,
        default=2,
        help="Maximum recursive depth for geo models.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input
    if input_path == DEFAULT_PREPARED_DATASET and not input_path.exists():
        input_path = ensure_example_dataset(input_path)

    try:
        summary = train_models(
            input_path=input_path,
            output_root=args.results_root,
            targets=args.targets,
            model_types=args.models,
            id_column=args.id_column,
            coordinate_columns=args.coord_columns,
            test_size=args.test_size,
            max_geo_depth=args.max_geo_depth,
        )
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Metrics saved to: {summary['metrics_path']}")
    print(f"Metadata saved to: {summary['metadata_path']}")
    for target, models in summary["trained_models"].items():
        print(f"{target}: {', '.join(models)}")
    if summary["skipped_models"]:
        print("Skipped models:")
        for model_name, reason in summary["skipped_models"].items():
            print(f"  {model_name}: {reason}")


if __name__ == "__main__":
    main()
