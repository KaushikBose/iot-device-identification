"""Test a trained RF IoT classifier on external labeled test files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from aggregation import AGGREGATION_MODES
from config import WINDOW_SIZE, LOCATIONS
from dataset import legacy_recordings, load_manifest
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import load_classes, load_locations, predict_signal
from model import StopGradientLayer  # noqa: F401 - registers layer for load_model


def wilson_confidence_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial accuracy."""
    if total <= 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt((proportion * (1 - proportion) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, float(center - margin))
    high = 1.0 if successes == total else min(1.0, float(center + margin))
    return low, high


def default_test_data_dir() -> Path:
    """Prefer data/tests, but support the existing data/test folder."""
    for candidate in (Path("data/tests"), Path("data/test")):
        if candidate.exists():
            return candidate
    return Path("data/tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_iot_classifier.h5"))
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV recording manifest. When provided, test the selected manifest split instead of --data-dir.",
    )
    parser.add_argument(
        "--manifest-split",
        default="test",
        help="Manifest split to test when --manifest is used (default: test).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/test"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_MODES,
        default="mean",
        help="How to combine per-window probabilities into each file prediction.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction of most confident windows used by top_confidence_mean aggregation.",
    )
    parser.add_argument(
        "--file-template",
        help="Optional file pattern such as '{class_name}_test.npy'. Defaults to auto-detecting raw and test names.",
    )
    parser.add_argument(
        "--require-all-classes",
        action="store_true",
        help="Fail if any class is missing from the test directory.",
    )
    return parser.parse_args()


def write_file_predictions(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write one row per test signal file."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "true_label",
                "predicted_label",
                "confidence",
                "true_location",
                "predicted_location",
                "location_confidence",
                "scenario",
                "capture_id",
                "split",
                "device_correct",
                "location_correct",
                "windows",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_test(
    model_path: Path,
    metadata_path: Path,
    data_dir: Path | None,
    output_dir: Path,
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
    file_template: str | None = None,
    allow_missing: bool = True,
    manifest_path: Path | None = None,
    manifest_split: str = "test",
    aggregation: str = "mean",
    top_fraction: float = 0.25,
) -> dict[str, object]:
    """Run file-level sliding-window ensemble tests and save reports."""
    classes = load_classes(metadata_path)
    locations = load_locations(metadata_path)
    if manifest_path:
        recordings = load_manifest(manifest_path, classes=classes, split=manifest_split)
        if not allow_missing:
            available = {recording.class_name for recording in recordings}
            missing = [class_name for class_name in classes if class_name not in available]
            if missing:
                raise ValueError(
                    f"Manifest split={manifest_split} has no recordings for classes: {', '.join(missing)}"
                )
    elif data_dir:
        recordings = legacy_recordings(
            data_dir,
            classes=classes,
            file_template=file_template,
            allow_missing=allow_missing,
        )
    else:
        raise ValueError("Provide data_dir or manifest_path")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path))
    rows: list[dict[str, object]] = []
    probabilities_by_file: dict[str, dict[str, object]] = {}
    
    y_dev_true: list[int] = []
    y_dev_pred: list[int] = []
    y_loc_true: list[int] = []
    y_loc_pred: list[int] = []

    for recording in recordings:
        signal = np.load(recording.path)
        result = predict_signal(
            model,
            signal,
            classes,
            locations=locations,
            window_size=window_size,
            step=step,
            aggregation=aggregation,
            top_fraction=top_fraction,
        )

        true_dev_idx = classes.index(recording.class_name)
        pred_dev_idx = classes.index(str(result["prediction"]))
        dev_correct = int(true_dev_idx == pred_dev_idx)

        # True location
        scenario = recording.scenario
        if scenario in locations:
            true_loc_idx = locations.index(scenario)
        else:
            true_loc_idx = -1
        
        pred_loc_idx = locations.index(str(result["location"]))
        loc_correct = int(true_loc_idx == pred_loc_idx)

        y_dev_true.append(true_dev_idx)
        y_dev_pred.append(pred_dev_idx)
        
        if true_loc_idx >= 0:
            y_loc_true.append(true_loc_idx)
            y_loc_pred.append(pred_loc_idx)

        probabilities_by_file[str(recording.path)] = {
            "device_probabilities": result["device_probabilities"],
            "location_probabilities": result["location_probabilities"],
        }
        
        rows.append(
            {
                "file": str(recording.path),
                "true_label": recording.class_name,
                "predicted_label": result["prediction"],
                "confidence": f"{float(result['device_confidence']):.8f}",
                "true_location": scenario,
                "predicted_location": result["location"],
                "location_confidence": f"{float(result['location_confidence']):.8f}",
                "scenario": scenario,
                "capture_id": recording.capture_id,
                "split": recording.split or "",
                "device_correct": dev_correct,
                "location_correct": loc_correct,
                "windows": int(result["windows"]),
            }
        )

    y_dev_true_array = np.asarray(y_dev_true, dtype=np.int64)
    y_dev_pred_array = np.asarray(y_dev_pred, dtype=np.int64)
    correct_dev_files = int(np.sum(y_dev_true_array == y_dev_pred_array))
    dev_accuracy = float(correct_dev_files / len(y_dev_true_array)) if len(y_dev_true_array) else 0.0
    dev_accuracy_ci_low, dev_accuracy_ci_high = wilson_confidence_interval(
        correct_dev_files,
        len(y_dev_true_array),
    )

    y_loc_true_array = np.asarray(y_loc_true, dtype=np.int64)
    y_loc_pred_array = np.asarray(y_loc_pred, dtype=np.int64)
    correct_loc_files = int(np.sum(y_loc_true_array == y_loc_pred_array))
    loc_accuracy = float(correct_loc_files / len(y_loc_true_array)) if len(y_loc_true_array) else 0.0
    loc_accuracy_ci_low, loc_accuracy_ci_high = wilson_confidence_interval(
        correct_loc_files,
        len(y_loc_true_array),
    )

    scenario_metrics = summarize_scenarios(rows)

    device_report = write_classification_report(
        y_dev_true_array,
        y_dev_pred_array,
        classes,
        output_dir / "device_classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_dev_true_array,
        y_dev_pred_array,
        classes,
        output_dir / "device_confusion_matrix.png",
    )

    location_report = write_classification_report(
        y_loc_true_array,
        y_loc_pred_array,
        locations,
        output_dir / "location_classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_loc_true_array,
        y_loc_pred_array,
        locations,
        output_dir / "location_confusion_matrix.png",
    )

    write_file_predictions(rows, output_dir / "predictions.csv")
    (output_dir / "probabilities.json").write_text(
        json.dumps(probabilities_by_file, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "evaluation_role": "external-file-level-ensemble",
        "evaluation_unit": "source_file",
        "data_dir": str(data_dir) if data_dir else None,
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_split": manifest_split if manifest_path else None,
        "model": str(model_path),
        "files": len(rows),
        "correct_device_files": correct_dev_files,
        "device_accuracy": dev_accuracy,
        "device_accuracy_ci_95_wilson": {
            "low": dev_accuracy_ci_low,
            "high": dev_accuracy_ci_high,
        },
        "correct_location_files": correct_loc_files,
        "location_accuracy": loc_accuracy,
        "location_accuracy_ci_95_wilson": {
            "low": loc_accuracy_ci_low,
            "high": loc_accuracy_ci_high,
        },
        "interpretation": "Each file contributes one prediction; window count is not the number of independent test examples.",
        "scenario_metrics": scenario_metrics,
        "window_size": int(window_size),
        "step": int(step),
        "aggregation": aggregation,
        "top_fraction": float(top_fraction),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    source = f"{manifest_path} split={manifest_split}" if manifest_path else str(data_dir)
    print(f"Tested {len(rows)} files from {source}")
    print(f"File-level Device Accuracy: {dev_accuracy:.4f}")
    print(f"File-level Location Accuracy: {loc_accuracy:.4f}")
    print("\n--- Device Classification Report ---")
    print(device_report)
    print("\n--- Location Classification Report ---")
    print(location_report)
    print(f"Saved reports to: {output_dir}")

    return metrics


def summarize_scenarios(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Summarize independent file-level outcomes for each scenario."""
    grouped_dev: dict[str, list[int]] = {}
    grouped_loc: dict[str, list[int]] = {}
    for row in rows:
        scenario = str(row.get("scenario") or "unspecified")
        correct_dev = row.get("device_correct", row.get("correct"))
        if correct_dev is not None:
            grouped_dev.setdefault(scenario, []).append(int(correct_dev))
        
        correct_loc = row.get("location_correct")
        if correct_loc is not None:
            grouped_loc.setdefault(scenario, []).append(int(correct_loc))

    summary: dict[str, dict[str, object]] = {}
    for scenario in sorted(set(grouped_dev.keys()).union(grouped_loc.keys())):
        dev_corr = grouped_dev.get(scenario, [])
        loc_corr = grouped_loc.get(scenario, [])
        files = max(len(dev_corr), len(loc_corr))
        
        entry = {"files": files}
        
        if dev_corr:
            dev_correct_files = int(sum(dev_corr))
            dev_low, dev_high = wilson_confidence_interval(dev_correct_files, len(dev_corr))
            if not loc_corr:
                entry["correct_files"] = dev_correct_files
                entry["accuracy"] = float(dev_correct_files / len(dev_corr))
                entry["accuracy_ci_95_wilson"] = {"low": dev_low, "high": dev_high}
            else:
                entry["device_correct_files"] = dev_correct_files
                entry["device_accuracy"] = float(dev_correct_files / len(dev_corr))
                entry["device_accuracy_ci_95_wilson"] = {"low": dev_low, "high": dev_high}
                
        if loc_corr:
            loc_correct_files = int(sum(loc_corr))
            loc_low, loc_high = wilson_confidence_interval(loc_correct_files, len(loc_corr))
            entry["location_correct_files"] = loc_correct_files
            entry["location_accuracy"] = float(loc_correct_files / len(loc_corr))
            entry["location_accuracy_ci_95_wilson"] = {"low": loc_low, "high": loc_high}
            
        summary[scenario] = entry
    return summary


def main() -> None:
    args = parse_args()
    run_test(
        model_path=args.model,
        metadata_path=args.metadata,
        data_dir=None if args.manifest else (args.data_dir or default_test_data_dir()),
        output_dir=args.output_dir,
        window_size=args.window_size,
        step=args.step,
        file_template=args.file_template,
        allow_missing=not args.require_all_classes,
        manifest_path=args.manifest,
        manifest_split=args.manifest_split,
        aggregation=args.aggregation,
        top_fraction=args.top_fraction,
    )


if __name__ == "__main__":
    main()
