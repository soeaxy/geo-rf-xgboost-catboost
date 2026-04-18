from __future__ import annotations

import csv
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ID_COLUMN = "采样点"
DEFAULT_SAMPLE_TARGET_FILES = {
    "Sn": REPO_ROOT / "Sn.xlsx",
    "Ta": REPO_ROOT / "Ta.xlsx",
}
DEFAULT_PREPARED_DATASET = REPO_ROOT / "data" / "input" / "example_dataset.csv"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_COORDINATE_CANDIDATES = (
    ("coord_x", "coord_y"),
    ("x", "y"),
    ("X", "Y"),
    ("lon", "lat"),
    ("longitude", "latitude"),
    ("POINT_X", "POINT_Y"),
)
DEFAULT_MODEL_ORDER = ("rf", "xgb", "cat")
XLSX_NAMESPACES = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


class DatasetError(RuntimeError):
    """Raised when an input dataset cannot be parsed or validated."""


@dataclass
class Dataset:
    path: Path
    columns: list[str]
    records: list[dict[str, Any]]
    spatial_frame: Any | None = None


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def safe_float(value: Any) -> float:
    number = coerce_scalar(value)
    if isinstance(number, (int, float, np.number)):
        return float(number)
    raise ValueError(f"Expected numeric value, got {value!r}")


def save_csv(path: str | Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    csv_path = Path(path)
    ensure_directory(csv_path.parent)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return csv_path


def load_csv_dataset(path: str | Path) -> Dataset:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        records: list[dict[str, Any]] = []
        for row in reader:
            parsed = {column: coerce_scalar(row.get(column)) for column in columns}
            if any(value is not None for value in parsed.values()):
                records.append(parsed)
    return Dataset(path=csv_path, columns=columns, records=records)


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    result = 0
    for char in letters:
        result = result * 26 + (ord(char.upper()) - 64)
    return result - 1


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    shared_strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in shared_strings_root.findall("main:si", XLSX_NAMESPACES):
        parts = [node.text or "" for node in item.iterfind(".//main:t", XLSX_NAMESPACES)]
        values.append("".join(parts))
    return values


def _resolve_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook_root.find("main:sheets", XLSX_NAMESPACES)
    if sheets is None:
        raise DatasetError("Workbook does not contain any sheets.")

    first_sheet = sheets.findall("main:sheet", XLSX_NAMESPACES)[0]
    relation_id = first_sheet.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    ]
    relationships_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationships = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships_root.findall("pkgrel:Relationship", XLSX_NAMESPACES)
    }
    target = relationships[relation_id].replace("\\", "/").lstrip("/")
    return f"xl/{target}"


def _parse_xlsx_cell(cell: ET.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.iterfind(".//main:t", XLSX_NAMESPACES)]
        return "".join(parts)

    value_node = cell.find("main:v", XLSX_NAMESPACES)
    if value_node is None:
        return None

    raw_value = value_node.text
    if raw_value is None:
        return None
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    if cell_type == "b":
        return 1 if raw_value == "1" else 0
    return raw_value


def load_xlsx_dataset(path: str | Path) -> Dataset:
    xlsx_path = Path(path)
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_root = ET.fromstring(archive.read(_resolve_sheet_path(archive)))
        sheet_data = sheet_root.find("main:sheetData", XLSX_NAMESPACES)
        if sheet_data is None:
            raise DatasetError(f"Worksheet is empty: {xlsx_path}")

        raw_rows: list[dict[int, Any]] = []
        max_columns = 0
        for row_node in sheet_data.findall("main:row", XLSX_NAMESPACES):
            row_values: dict[int, Any] = {}
            for cell in row_node.findall("main:c", XLSX_NAMESPACES):
                cell_ref = cell.attrib.get("r", "")
                column_index = _column_index(cell_ref)
                row_values[column_index] = _parse_xlsx_cell(cell, shared_strings)
                max_columns = max(max_columns, column_index + 1)
            raw_rows.append(row_values)

    if not raw_rows:
        raise DatasetError(f"Worksheet is empty: {xlsx_path}")

    headers: list[str] = []
    for index in range(max_columns):
        header_value = raw_rows[0].get(index)
        header = str(header_value).strip() if header_value is not None else ""
        headers.append(header or f"column_{index + 1}")

    records: list[dict[str, Any]] = []
    for row_values in raw_rows[1:]:
        record = {header: coerce_scalar(row_values.get(index)) for index, header in enumerate(headers)}
        if any(value is not None for value in record.values()):
            records.append(record)

    return Dataset(path=xlsx_path, columns=headers, records=records)


def load_shapefile_dataset(path: str | Path) -> Dataset:
    shp_path = Path(path)
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise DatasetError(
            "Shapefile input requires geopandas. Install the optional spatial dependencies first."
        ) from exc

    gdf = gpd.read_file(shp_path)
    columns = [column for column in gdf.columns if column != "geometry"]
    has_point_geometry = hasattr(gdf.geometry, "x") and hasattr(gdf.geometry, "y")
    if has_point_geometry:
        if "coord_x" not in columns:
            columns.append("coord_x")
        if "coord_y" not in columns:
            columns.append("coord_y")

    records: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        record = {column: coerce_scalar(row[column]) for column in gdf.columns if column != "geometry"}
        if has_point_geometry:
            record["coord_x"] = float(row.geometry.x)
            record["coord_y"] = float(row.geometry.y)
        records.append(record)

    return Dataset(path=shp_path, columns=columns, records=records, spatial_frame=gdf)


def load_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return load_csv_dataset(dataset_path)
    if suffix in {".xlsx", ".xlsm"}:
        return load_xlsx_dataset(dataset_path)
    if suffix == ".shp":
        return load_shapefile_dataset(dataset_path)
    raise DatasetError(f"Unsupported dataset format: {dataset_path.suffix}")


def prepare_example_dataset(
    output_path: str | Path = DEFAULT_PREPARED_DATASET,
    sample_target_files: dict[str, str | Path] | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
) -> Path:
    output_csv = Path(output_path)
    sources = sample_target_files or DEFAULT_SAMPLE_TARGET_FILES
    prepared_sources = {target: Path(path) for target, path in sources.items()}

    if not prepared_sources:
        raise DatasetError("No sample files were provided.")

    source_items = list(prepared_sources.items())
    base_target, base_path = source_items[0]
    base_dataset = load_dataset(base_path)
    if id_column not in base_dataset.columns:
        raise DatasetError(f"{base_path.name} is missing id column {id_column!r}.")
    if base_target not in base_dataset.columns:
        raise DatasetError(f"{base_path.name} is missing target column {base_target!r}.")

    feature_columns = [
        column for column in base_dataset.columns if column not in {id_column, base_target}
    ]
    ordered_ids: list[str] = []
    merged_rows: dict[str, dict[str, Any]] = {}

    for record in base_dataset.records:
        sample_id = str(record[id_column])
        ordered_ids.append(sample_id)
        merged_rows[sample_id] = {
            id_column: sample_id,
            base_target: safe_float(record[base_target]),
        }
        for feature in feature_columns:
            merged_rows[sample_id][feature] = record.get(feature)

    for target, source_path in source_items[1:]:
        dataset = load_dataset(source_path)
        if id_column not in dataset.columns:
            raise DatasetError(f"{source_path.name} is missing id column {id_column!r}.")
        if target not in dataset.columns:
            raise DatasetError(f"{source_path.name} is missing target column {target!r}.")

        for record in dataset.records:
            sample_id = str(record[id_column])
            if sample_id not in merged_rows:
                merged_rows[sample_id] = {id_column: sample_id}
                ordered_ids.append(sample_id)
            merged_rows[sample_id][target] = safe_float(record[target])
            for feature in feature_columns:
                if feature not in merged_rows[sample_id]:
                    merged_rows[sample_id][feature] = record.get(feature)

    fieldnames = [id_column] + list(prepared_sources.keys()) + feature_columns
    rows = [merged_rows[sample_id] for sample_id in ordered_ids]
    return save_csv(output_csv, fieldnames, rows)


def ensure_example_dataset(output_path: str | Path = DEFAULT_PREPARED_DATASET) -> Path:
    output_csv = Path(output_path)
    if output_csv.exists():
        return output_csv
    return prepare_example_dataset(output_csv)


def detect_target_columns(columns: Sequence[str], requested_targets: Sequence[str] | None = None) -> list[str]:
    if requested_targets:
        missing = [target for target in requested_targets if target not in columns]
        if missing:
            raise DatasetError(f"Missing target columns: {', '.join(missing)}")
        return list(requested_targets)

    inferred = [target for target in DEFAULT_SAMPLE_TARGET_FILES if target in columns]
    if not inferred:
        raise DatasetError(
            "Could not infer any targets. Pass --targets explicitly or prepare the bundled sample data first."
        )
    return inferred


def detect_coordinate_columns(
    columns: Sequence[str],
    requested_coordinate_columns: Sequence[str] | None = None,
) -> tuple[str, str] | None:
    if requested_coordinate_columns:
        if len(requested_coordinate_columns) != 2:
            raise DatasetError("Coordinate columns must contain exactly two column names.")
        x_column, y_column = requested_coordinate_columns
        if x_column not in columns or y_column not in columns:
            raise DatasetError(f"Coordinate columns not found: {x_column}, {y_column}")
        return x_column, y_column

    for x_column, y_column in DEFAULT_COORDINATE_CANDIDATES:
        if x_column in columns and y_column in columns:
            return x_column, y_column
    return None


def is_numeric_column(records: Sequence[dict[str, Any]], column: str) -> bool:
    saw_value = False
    for record in records:
        value = record.get(column)
        if value is None:
            continue
        saw_value = True
        if not isinstance(coerce_scalar(value), (int, float, np.number)):
            return False
    return saw_value


def infer_feature_columns(
    dataset: Dataset,
    target_columns: Sequence[str],
    id_column: str,
    coordinate_columns: Sequence[str] | None,
) -> list[str]:
    excluded = {id_column, *target_columns}
    if coordinate_columns:
        excluded.update(coordinate_columns)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in excluded and is_numeric_column(dataset.records, column)
    ]
    if not feature_columns:
        raise DatasetError("No numeric feature columns were detected in the dataset.")
    return feature_columns


def build_learning_arrays(
    dataset: Dataset,
    feature_columns: Sequence[str],
    target_column: str,
    id_column: str,
    coordinate_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    sample_ids: list[str] = []
    features: list[list[float]] = []
    targets: list[float] = []
    coordinates: list[list[float]] = []

    for record in dataset.records:
        try:
            feature_values = [safe_float(record[column]) for column in feature_columns]
            target_value = safe_float(record[target_column])
            if coordinate_columns:
                coordinate_values = [safe_float(record[column]) for column in coordinate_columns]
            else:
                coordinate_values = []
        except (TypeError, ValueError):
            continue

        sample_ids.append(str(record.get(id_column)))
        features.append(feature_values)
        targets.append(target_value)
        if coordinate_values:
            coordinates.append(coordinate_values)

    if not features:
        raise DatasetError(f"No usable rows were found for target {target_column!r}.")

    feature_array = np.asarray(features, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float) if coordinates else None
    return np.asarray(sample_ids, dtype=object), feature_array, target_array, coordinate_array


def build_model_factories(
    random_state: int = 42,
    rf_estimators: int = 200,
    boost_iterations: int = 300,
) -> tuple[dict[str, Callable[[], Any]], dict[str, str]]:
    factories: dict[str, Callable[[], Any]] = {
        "rf": lambda: RandomForestRegressor(
            n_estimators=rf_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    }
    unavailable: dict[str, str] = {}

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        unavailable["xgb"] = f"xgboost is not installed ({exc})."
    else:
        factories["xgb"] = lambda: XGBRegressor(
            n_estimators=boost_iterations,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        unavailable["cat"] = f"catboost is not installed ({exc})."
    else:
        factories["cat"] = lambda: CatBoostRegressor(
            iterations=boost_iterations,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
        )

    return factories, unavailable


def fit_geo_model(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    model_factory: Callable[[], Any],
    random_state: int,
    max_depth: int = 2,
    min_leaf_size: int = 25,
    depth: int = 0,
) -> dict[str, Any]:
    if depth >= max_depth or len(y) < max(min_leaf_size * 2, 10):
        model = model_factory()
        model.fit(X, y)
        return {"model": model}

    if np.allclose(coords, coords[0]):
        model = model_factory()
        model.fit(X, y)
        return {"model": model}

    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        model = model_factory()
        model.fit(X, y)
        return {"model": model}

    partitions: dict[int, Any] = {}
    for label in unique_labels:
        label_mask = labels == label
        partitions[int(label)] = fit_geo_model(
            X[label_mask],
            y[label_mask],
            coords[label_mask],
            model_factory=model_factory,
            random_state=random_state,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            depth=depth + 1,
        )
    return {"kmeans": kmeans, "partitions": partitions}


def predict_geo_model(model_dict: dict[str, Any], X: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if "model" in model_dict:
        return np.asarray(model_dict["model"].predict(X), dtype=float)

    labels = model_dict["kmeans"].predict(coords)
    predictions = np.zeros(X.shape[0], dtype=float)
    for label in np.unique(labels):
        label_mask = labels == label
        predictions[label_mask] = predict_geo_model(
            model_dict["partitions"][int(label)],
            X[label_mask],
            coords[label_mask],
        )
    return predictions


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "sample_count": float(len(y_true)),
    }


def extract_feature_importances(model_or_tree: Any) -> np.ndarray | None:
    if isinstance(model_or_tree, dict):
        collected: list[np.ndarray] = []
        if "model" in model_or_tree:
            importance = extract_feature_importances(model_or_tree["model"])
            if importance is not None:
                collected.append(importance)
        else:
            for subtree in model_or_tree["partitions"].values():
                importance = extract_feature_importances(subtree)
                if importance is not None:
                    collected.append(importance)
        if not collected:
            return None
        return np.mean(np.vstack(collected), axis=0)

    if hasattr(model_or_tree, "feature_importances_"):
        return np.asarray(model_or_tree.feature_importances_, dtype=float)
    if hasattr(model_or_tree, "get_feature_importance"):
        return np.asarray(model_or_tree.get_feature_importance(), dtype=float)
    return None


def save_feature_importance_artifacts(
    target_column: str,
    model_name: str,
    importances: np.ndarray | None,
    feature_columns: Sequence[str],
    output_directory: str | Path,
) -> tuple[Path, Path] | None:
    if importances is None:
        return None

    importance_dir = ensure_directory(output_directory)
    ranked_pairs = sorted(
        zip(feature_columns, importances.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    csv_path = importance_dir / f"{target_column}_{model_name}_feature_importance.csv"
    save_csv(
        csv_path,
        ["feature", "importance"],
        [{"feature": feature, "importance": importance} for feature, importance in ranked_pairs],
    )

    top_pairs = ranked_pairs[: min(20, len(ranked_pairs))]
    plot_path = importance_dir / f"{target_column}_{model_name}_feature_importance.png"
    labels = [feature for feature, _ in reversed(top_pairs)]
    values = [importance for _, importance in reversed(top_pairs)]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Importance")
    plt.title(f"{target_column} - {model_name} feature importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return csv_path, plot_path


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    json_path = Path(path)
    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def train_models(
    input_path: str | Path = DEFAULT_PREPARED_DATASET,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
    targets: Sequence[str] | None = None,
    model_types: Sequence[str] | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    coordinate_columns: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_geo_depth: int = 2,
    min_geo_leaf_size: int = 25,
    rf_estimators: int = 200,
    boost_iterations: int = 300,
) -> dict[str, Any]:
    dataset = load_dataset(input_path)
    target_columns = detect_target_columns(dataset.columns, targets)
    active_coordinate_columns = detect_coordinate_columns(dataset.columns, coordinate_columns)
    feature_columns = infer_feature_columns(
        dataset,
        target_columns=target_columns,
        id_column=id_column,
        coordinate_columns=active_coordinate_columns,
    )

    factories, unavailable_models = build_model_factories(
        random_state=random_state,
        rf_estimators=rf_estimators,
        boost_iterations=boost_iterations,
    )
    requested_models = list(model_types or DEFAULT_MODEL_ORDER)
    active_models = [model_name for model_name in requested_models if model_name in factories]
    skipped_models = {
        model_name: unavailable_models[model_name]
        for model_name in requested_models
        if model_name in unavailable_models
    }
    if not active_models:
        raise DatasetError("No requested models are available in the current Python environment.")

    results_root = ensure_directory(output_root)
    model_dir = ensure_directory(results_root / "models")
    metrics_dir = ensure_directory(results_root / "metrics")
    prediction_dir = ensure_directory(results_root / "predictions")
    metadata_dir = ensure_directory(results_root / "metadata")
    feature_importance_dir = ensure_directory(results_root / "feature_importance")

    metrics_rows: list[dict[str, Any]] = []
    trained_models: dict[str, list[str]] = {}
    holdout_paths: dict[str, str] = {}

    for target_column in target_columns:
        sample_ids, X, y, coords = build_learning_arrays(
            dataset,
            feature_columns=feature_columns,
            target_column=target_column,
            id_column=id_column,
            coordinate_columns=active_coordinate_columns,
        )
        if len(y) < 10:
            raise DatasetError(f"Target {target_column!r} has too few usable rows ({len(y)}).")

        indices = np.arange(len(y))
        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        holdout_rows = {
            int(index): {id_column: str(sample_ids[index]), target_column: float(y[index])}
            for index in test_index.tolist()
        }
        trained_models[target_column] = []

        for model_name in active_models:
            estimator = factories[model_name]()
            estimator.fit(X[train_index], y[train_index])
            predictions = np.asarray(estimator.predict(X[test_index]), dtype=float)
            metrics = evaluate_predictions(y[test_index], predictions)
            metrics_rows.append(
                {
                    "target": target_column,
                    "model": model_name,
                    **metrics,
                }
            )
            dump(estimator, model_dir / f"{target_column}_{model_name}_model.joblib")
            trained_models[target_column].append(model_name)
            for index, prediction in zip(test_index.tolist(), predictions.tolist()):
                holdout_rows[int(index)][f"{target_column}_{model_name}"] = prediction
            save_feature_importance_artifacts(
                target_column,
                model_name,
                extract_feature_importances(estimator),
                feature_columns,
                feature_importance_dir,
            )

            if coords is not None:
                geo_name = f"geo_{model_name}"
                geo_estimator = fit_geo_model(
                    X[train_index],
                    y[train_index],
                    coords[train_index],
                    model_factory=factories[model_name],
                    random_state=random_state,
                    max_depth=max_geo_depth,
                    min_leaf_size=min_geo_leaf_size,
                )
                geo_predictions = predict_geo_model(geo_estimator, X[test_index], coords[test_index])
                geo_metrics = evaluate_predictions(y[test_index], geo_predictions)
                metrics_rows.append(
                    {
                        "target": target_column,
                        "model": geo_name,
                        **geo_metrics,
                    }
                )
                dump(geo_estimator, model_dir / f"{target_column}_{geo_name}_model.joblib")
                trained_models[target_column].append(geo_name)
                for index, prediction in zip(test_index.tolist(), geo_predictions.tolist()):
                    holdout_rows[int(index)][f"{target_column}_{geo_name}"] = prediction
                save_feature_importance_artifacts(
                    target_column,
                    geo_name,
                    extract_feature_importances(geo_estimator),
                    feature_columns,
                    feature_importance_dir,
                )

        holdout_fieldnames = [id_column, target_column] + [
            f"{target_column}_{model_name}" for model_name in trained_models[target_column]
        ]
        holdout_path = prediction_dir / f"holdout_{target_column}.csv"
        save_csv(holdout_path, holdout_fieldnames, list(holdout_rows.values()))
        holdout_paths[target_column] = str(holdout_path)

    metrics_path = metrics_dir / "model_metrics.csv"
    save_csv(
        metrics_path,
        ["target", "model", "r2", "rmse", "mae", "bias", "sample_count"],
        metrics_rows,
    )

    metadata = {
        "input_path": str(Path(input_path).resolve()),
        "id_column": id_column,
        "targets": target_columns,
        "feature_columns": feature_columns,
        "coordinate_columns": list(active_coordinate_columns) if active_coordinate_columns else None,
        "trained_models": trained_models,
        "skipped_models": skipped_models,
        "holdout_predictions": holdout_paths,
        "results_root": str(results_root.resolve()),
    }
    metadata_path = save_json(metadata_dir / "training_metadata.json", metadata)

    return {
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
        "model_dir": str(model_dir),
        "prediction_dir": str(prediction_dir),
        "trained_models": trained_models,
        "skipped_models": skipped_models,
        "coordinate_columns": list(active_coordinate_columns) if active_coordinate_columns else None,
        "feature_columns": feature_columns,
    }


def load_training_metadata(metadata_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def predict_with_saved_models(
    input_path: str | Path,
    output_root: str | Path = DEFAULT_RESULTS_ROOT,
    metadata_path: str | Path | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    results_root = ensure_directory(output_root)
    metadata = load_training_metadata(
        metadata_path or results_root / "metadata" / "training_metadata.json"
    )
    dataset = load_dataset(input_path)
    feature_columns = metadata["feature_columns"]
    target_columns = metadata["targets"]
    id_column = metadata["id_column"]
    coordinate_columns = metadata.get("coordinate_columns")
    trained_models: dict[str, list[str]] = metadata["trained_models"]

    missing_features = [column for column in feature_columns if column not in dataset.columns]
    if missing_features:
        raise DatasetError(
            f"Prediction input is missing required feature columns: {', '.join(missing_features)}"
        )
    if id_column not in dataset.columns:
        raise DatasetError(f"Prediction input is missing id column {id_column!r}.")

    model_dir = results_root / "models"
    output_rows = []
    for record in dataset.records:
        row = {id_column: record.get(id_column)}
        for target_column in target_columns:
            if target_column in dataset.columns:
                row[target_column] = record.get(target_column)
        output_rows.append(row)

    output_columns = [id_column] + [target for target in target_columns if target in dataset.columns]
    cached_features: dict[int, list[float]] = {}
    cached_coords: dict[int, list[float]] = {}

    for row_index, record in enumerate(dataset.records):
        try:
            cached_features[row_index] = [safe_float(record[column]) for column in feature_columns]
        except (TypeError, ValueError):
            continue
        if coordinate_columns:
            try:
                cached_coords[row_index] = [safe_float(record[column]) for column in coordinate_columns]
            except (TypeError, ValueError):
                pass

    for target_column, model_names in trained_models.items():
        for model_name in model_names:
            model_path = model_dir / f"{target_column}_{model_name}_model.joblib"
            if not model_path.exists():
                continue
            available_indices: list[int] = []
            features: list[list[float]] = []
            coordinates: list[list[float]] = []
            for row_index in range(len(dataset.records)):
                if row_index not in cached_features:
                    continue
                if model_name.startswith("geo_"):
                    if not coordinate_columns or row_index not in cached_coords:
                        continue
                    coordinates.append(cached_coords[row_index])
                available_indices.append(row_index)
                features.append(cached_features[row_index])

            if not available_indices:
                continue

            estimator = load(model_path)
            feature_array = np.asarray(features, dtype=float)
            if model_name.startswith("geo_"):
                coordinate_array = np.asarray(coordinates, dtype=float)
                predictions = predict_geo_model(estimator, feature_array, coordinate_array)
            else:
                predictions = np.asarray(estimator.predict(feature_array), dtype=float)

            output_column = f"{target_column}_{model_name}"
            if output_column not in output_columns:
                output_columns.append(output_column)
            for row_index, prediction in zip(available_indices, predictions.tolist()):
                output_rows[row_index][output_column] = prediction

    prediction_dir = ensure_directory(results_root / "predictions")
    prediction_name = output_name or f"{Path(input_path).stem}_predictions.csv"
    prediction_path = save_csv(prediction_dir / prediction_name, output_columns, output_rows)

    shapefile_path = None
    if dataset.spatial_frame is not None:
        try:
            spatial_frame = dataset.spatial_frame.copy()
            for row in output_rows:
                sample_id = row[id_column]
                row_mask = spatial_frame[id_column].astype(str) == str(sample_id)
                for column in output_columns:
                    if column == id_column:
                        continue
                    if column in row:
                        spatial_frame.loc[row_mask, column] = row[column]
            shapefile_path = prediction_dir / f"{Path(input_path).stem}_predictions.shp"
            spatial_frame.to_file(shapefile_path, encoding="utf-8")
        except Exception:
            shapefile_path = None

    return {
        "prediction_path": str(prediction_path),
        "shapefile_path": str(shapefile_path) if shapefile_path else None,
    }

