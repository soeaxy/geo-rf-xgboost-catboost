from __future__ import annotations

import argparse
from pathlib import Path

from geo_pipeline import (
    DEFAULT_ID_COLUMN,
    DEFAULT_PREPARED_DATASET,
    DEFAULT_SAMPLE_TARGET_FILES,
    prepare_example_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a tabular example dataset from the bundled Sn/Ta workbooks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREPARED_DATASET,
        help="Where to save the merged example dataset CSV.",
    )
    parser.add_argument(
        "--sn",
        type=Path,
        default=DEFAULT_SAMPLE_TARGET_FILES["Sn"],
        help="Path to the Sn workbook.",
    )
    parser.add_argument(
        "--ta",
        type=Path,
        default=DEFAULT_SAMPLE_TARGET_FILES["Ta"],
        help="Path to the Ta workbook.",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help="Sample id column shared by all input workbooks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = prepare_example_dataset(
        output_path=args.output,
        sample_target_files={"Sn": args.sn, "Ta": args.ta},
        id_column=args.id_column,
    )
    print(f"Prepared example dataset: {output_path}")


if __name__ == "__main__":
    main()
