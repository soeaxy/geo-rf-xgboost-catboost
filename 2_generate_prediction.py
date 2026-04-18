from __future__ import annotations

import argparse
from pathlib import Path

from geo_pipeline import (
    DEFAULT_PREPARED_DATASET,
    DEFAULT_RESULTS_ROOT,
    DatasetError,
    ensure_example_dataset,
    predict_with_saved_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate predictions for a new dataset using the trained models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PREPARED_DATASET,
        help="Prediction input (.csv, .xlsx, or .shp). Defaults to the prepared example dataset.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing the trained models and metadata.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional training metadata path. Defaults to results/metadata/training_metadata.json.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output CSV file name inside results/predictions.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input
    if input_path == DEFAULT_PREPARED_DATASET and not input_path.exists():
        input_path = ensure_example_dataset(input_path)

    try:
        summary = predict_with_saved_models(
            input_path=input_path,
            output_root=args.results_root,
            metadata_path=args.metadata,
            output_name=args.output_name,
        )
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Predictions saved to: {summary['prediction_path']}")
    if summary["shapefile_path"]:
        print(f"Spatial predictions saved to: {summary['shapefile_path']}")


if __name__ == "__main__":
    main()
