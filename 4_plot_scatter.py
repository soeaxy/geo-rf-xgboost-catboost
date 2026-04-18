from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_pipeline import DEFAULT_RESULTS_ROOT, DatasetError, evaluate_predictions, load_csv_dataset


def calculate_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = evaluate_predictions(y_true, y_pred)
    metrics["rpd"] = float(np.std(y_true) / metrics["rmse"]) if metrics["rmse"] else float("nan")
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    metrics["lccc"] = float(
        (2 * correlation * np.std(y_true) * np.std(y_pred))
        / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one regression scatter chart.")
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
        "--model-column",
        default=None,
        help="Prediction column to plot. Defaults to the first <target>_* column in the CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to figures/<model-column>_scatter.png.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or (DEFAULT_RESULTS_ROOT / "predictions" / f"holdout_{args.target}.csv")
    dataset = load_csv_dataset(input_path)

    if args.target not in dataset.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    model_column = args.model_column
    if model_column is None:
        candidates = [
            column
            for column in dataset.columns
            if column.startswith(f"{args.target}_") and column != args.target
        ]
        if not candidates:
            raise SystemExit(f"No prediction columns found for target {args.target}.")
        model_column = candidates[0]

    if model_column not in dataset.columns:
        raise SystemExit(f"Prediction column not found: {model_column}")

    pairs = []
    for record in dataset.records:
        try:
            pairs.append((float(record[args.target]), float(record[model_column])))
        except (TypeError, ValueError):
            continue
    if not pairs:
        raise SystemExit("No valid rows were found for plotting.")

    y_true = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_pred = np.asarray([pair[1] for pair in pairs], dtype=float)
    metrics = calculate_extended_metrics(y_true, y_pred)

    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))
    line_x = np.linspace(min_value, max_value, 100)
    if len(np.unique(y_true)) > 1:
        slope, intercept = np.polyfit(y_true, y_pred, deg=1)
        trend_y = slope * line_x + intercept
    else:
        trend_y = np.full_like(line_x, np.mean(y_pred))

    output_path = args.output or (Path("figures") / f"{model_column}_scatter.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, s=28)
    plt.plot(line_x, trend_y, color="tab:red", linewidth=2, label="Trend")
    plt.plot(line_x, line_x, color="black", linestyle="--", linewidth=1.5, label="1:1")
    metrics_text = (
        f"R2 = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"RPD = {metrics['rpd']:.3f}\n"
        f"LCCC = {metrics['lccc']:.3f}\n"
        f"Bias = {metrics['bias']:.3f}"
    )
    plt.text(
        0.04,
        0.96,
        metrics_text,
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    plt.xlabel(f"Measured {args.target}")
    plt.ylabel(f"Predicted {args.target}")
    plt.title(f"{args.target} - {model_column}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to: {output_path}")


if __name__ == "__main__":
    main()
