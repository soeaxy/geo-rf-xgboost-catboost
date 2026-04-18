from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from geo_pipeline import load_dataset, predict_with_saved_models, prepare_example_dataset, train_models


class GeoPipelineTestCase(unittest.TestCase):
    def test_prepare_example_dataset_merges_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "example_dataset.csv"
            prepare_example_dataset(output_path=output_path)
            dataset = load_dataset(output_path)

            self.assertIn("采样点", dataset.columns)
            self.assertIn("Sn", dataset.columns)
            self.assertIn("Ta", dataset.columns)
            self.assertIn("VV", dataset.columns)
            self.assertEqual(len(dataset.records), 104)
            self.assertEqual(dataset.records[0]["采样点"], "XF-S01-1")

    def test_train_and_predict_rf_on_example_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            data_path = temp_root / "example_dataset.csv"
            prepare_example_dataset(output_path=data_path)

            results_root = temp_root / "results"
            summary = train_models(
                input_path=data_path,
                output_root=results_root,
                targets=["Sn", "Ta"],
                model_types=["rf"],
                rf_estimators=80,
            )

            self.assertTrue((results_root / "models" / "Sn_rf_model.joblib").exists())
            self.assertTrue((results_root / "models" / "Ta_rf_model.joblib").exists())
            self.assertIn("Sn", summary["trained_models"])
            self.assertEqual(summary["trained_models"]["Sn"], ["rf"])

            metrics_path = results_root / "metrics" / "model_metrics.csv"
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)

            prediction_summary = predict_with_saved_models(
                input_path=data_path,
                output_root=results_root,
                output_name="example_predictions.csv",
            )
            prediction_dataset = load_dataset(prediction_summary["prediction_path"])
            self.assertIn("Sn_rf", prediction_dataset.columns)
            self.assertIn("Ta_rf", prediction_dataset.columns)
            self.assertEqual(len(prediction_dataset.records), 104)


if __name__ == "__main__":
    unittest.main()
