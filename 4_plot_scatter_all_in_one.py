from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_pipeline import DEFAULT_RESULTS_ROOT, evaluate_predictions, load_csv_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot all available prediction columns for one target in a single figure."
    )
    parser.add_argument(
        "--target",
        default="Sn",
        help="Target name used to resolve the default holdout prediction file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input prediction CSV. Defaults to results/predictions/holdout_<target>.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to figures/<target>_all_models_scatter.png.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or (DEFAULT_RESULTS_ROOT / "predictions" / f"holdout_{args.target}.csv")
    dataset = load_csv_dataset(input_path)

    if args.target not in dataset.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    model_columns = [
        column for column in dataset.columns if column.startswith(f"{args.target}_") and column != args.target
    ]
    if not model_columns:
        raise SystemExit(f"No prediction columns found for target {args.target}.")

    pairs_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_column in model_columns:
        pairs = []
        for record in dataset.records:
            try:
                pairs.append((float(record[args.target]), float(record[model_column])))
            except (TypeError, ValueError):
                continue
        if pairs:
            y_true = np.asarray([pair[0] for pair in pairs], dtype=float)
            y_pred = np.asarray([pair[1] for pair in pairs], dtype=float)
            pairs_by_model[model_column] = (y_true, y_pred)

    if not pairs_by_model:
        raise SystemExit("No valid rows were found for plotting.")

    total_plots = len(pairs_by_model)
    columns = min(3, total_plots)
    rows = math.ceil(total_plots / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    for axis in axes_array.ravel():
        axis.set_visible(False)

    for axis, (model_column, (y_true, y_pred)) in zip(axes_array.ravel(), pairs_by_model.items()):
        axis.set_visible(True)
        min_value = min(np.min(y_true), np.min(y_pred))
        max_value = max(np.max(y_true), np.max(y_pred))
        line_x = np.linspace(min_value, max_value, 100)
        if len(np.unique(y_true)) > 1:
            slope, intercept = np.polyfit(y_true, y_pred, deg=1)
            trend_y = slope * line_x + intercept
        else:
            trend_y = np.full_like(line_x, np.mean(y_pred))
        metrics = evaluate_predictions(y_true, y_pred)

        axis.scatter(y_true, y_pred, alpha=0.75, s=24)
        axis.plot(line_x, trend_y, color="tab:red", linewidth=2)
        axis.plot(line_x, line_x, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(model_column)
        axis.set_xlabel(f"Measured {args.target}")
        axis.set_ylabel(f"Predicted {args.target}")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.text(
            0.03,
            0.97,
            f"R2={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.3f}\nMAE={metrics['mae']:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

    output_path = args.output or (Path("figures") / f"{args.target}_all_models_scatter.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Scatter summary saved to: {output_path}")


if __name__ == "__main__":
    main()
