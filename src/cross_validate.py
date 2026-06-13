"""Cross-validation runner for Leave-One-Recording-Out (LORO) and Leave-One-Environment-Out (LOEO)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from aggregation import AGGREGATION_MODES
from config import BATCH_SIZE, CLASSES, EPOCHS, NUM_CLASSES, NUM_LOCATIONS, SEED, LOCATIONS
from dataset import build_dataset_from_recordings, load_manifest
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import load_classes, load_locations, predict_signal
from model import build_cnn, StopGradientLayer  # noqa: F401 - registers layer for load_model
from test import wilson_confidence_interval, write_file_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/recordings_manifest.csv"),
        help="CSV recording manifest with all scenario captures.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["loeo", "loro-group", "loro"],
        default="loeo",
        help="Cross-validation mode: loeo (leave-one-environment-out), loro-group (leave-one-scenario-capture-group-out), or loro (leave-one-recording-out).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["avg", "max", "avgmax", "flatten"],
        default="avg",
        help="Pooling layer to use before classification head.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_MODES,
        default="mean",
        help="How to combine per-window probabilities into file-level predictions.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction of most confident windows used by top_confidence_mean aggregation.",
    )
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Optionally sample at most this many windows per class in each split before spectrogram conversion.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/cv"))
    return parser.parse_args()


def get_folds(recordings: list, mode: str) -> list[tuple[str, list, list]]:
    """Determine the train/test splits for cross validation based on mode.

    Returns:
        List of tuples: (fold_name, train_recordings, test_recordings)
    """
    folds = []
    if mode == "loeo":
        # Leave-One-Environment-Out: group by scenario
        scenarios = sorted(list({r.scenario for r in recordings}))
        for scenario in scenarios:
            test_recs = [r for r in recordings if r.scenario == scenario]
            train_recs = [r for r in recordings if r.scenario != scenario]
            folds.append((f"LOEO_{scenario}", train_recs, test_recs))
    elif mode == "loro-group":
        # Leave-One-Group-Out: group by (scenario, capture_id)
        groups = sorted(list({(r.scenario, r.capture_id) for r in recordings}))
        for scenario, capture_id in groups:
            group_name = f"{scenario}_cap{capture_id}"
            test_recs = [
                r for r in recordings if r.scenario == scenario and r.capture_id == capture_id
            ]
            train_recs = [
                r for r in recordings if not (r.scenario == scenario and r.capture_id == capture_id)
            ]
            folds.append((f"LORO_Group_{group_name}", train_recs, test_recs))
    elif mode == "loro":
        # True Leave-One-Recording-Out: group by individual recording path
        for idx, rec in enumerate(recordings):
            fold_name = f"LORO_File_{rec.path.stem}"
            test_recs = [rec]
            train_recs = [r for r in recordings if r.path != rec.path]
            folds.append((fold_name, train_recs, test_recs))
    return folds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    recordings = load_manifest(args.manifest, classes=CLASSES)
    locations = LOCATIONS
    folds = get_folds(recordings, args.mode)
    print(f"Starting Cross-Validation (Mode: {args.mode.upper()}) with {len(folds)} folds.")
    print(f"Architecture configuration: pooling={args.pooling}, epochs={args.epochs}")

    fold_dev_accuracies: list[float] = []
    fold_loc_accuracies: list[float] = []
    all_predictions_rows: list[dict[str, object]] = []
    
    y_dev_true_all: list[int] = []
    y_dev_pred_all: list[int] = []
    y_loc_true_all: list[int] = []
    y_loc_pred_all: list[int] = []

    for idx, (fold_name, train_recs, test_recs) in enumerate(folds, start=1):
        print(f"\n--- Fold {idx}/{len(folds)}: {fold_name} ---")
        print(f"Training on {len(train_recs)} recordings. Testing on {len(test_recs)} recordings.")

        # Build training / validation window datasets
        # We extract all windows from train_recs and split them 85% train, 15% val
        x_train_val, y_dev_train_val, y_loc_train_val = build_dataset_from_recordings(
            train_recs,
            classes=CLASSES,
            locations=locations,
            balance=not args.no_balance,
            seed=args.seed,
            max_windows_per_class=args.max_windows_per_class,
        )

        (
            x_train, x_val,
            y_dev_train, y_dev_val,
            y_loc_train, y_loc_val,
        ) = train_test_split(
            x_train_val,
            y_dev_train_val,
            y_loc_train_val,
            test_size=0.15,
            stratify=y_dev_train_val,
            random_state=args.seed,
        )

        y_dev_train_cat = tf.keras.utils.to_categorical(y_dev_train, NUM_CLASSES)
        y_dev_val_cat = tf.keras.utils.to_categorical(y_dev_val, NUM_CLASSES)
        y_loc_train_cat = tf.keras.utils.to_categorical(y_loc_train, NUM_LOCATIONS)
        y_loc_val_cat = tf.keras.utils.to_categorical(y_loc_val, NUM_LOCATIONS)

        # Build and compile model
        model = build_cnn(
            input_shape=x_train.shape[1:],
            num_classes=NUM_CLASSES,
            num_locations=NUM_LOCATIONS,
            learning_rate=args.learning_rate,
            pooling=args.pooling,
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor="val_device_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode="min",
                verbose=0,
            ),
            EarlyStopping(
                monitor="val_device_accuracy",
                mode="max",
                patience=6,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        # Fit model
        model.fit(
            x_train,
            {"device": y_dev_train_cat, "location": y_loc_train_cat},
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(
                x_val,
                {"device": y_dev_val_cat, "location": y_loc_val_cat},
            ),
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on test recordings at the file (ensemble) level
        correct_dev_fold = 0
        correct_loc_fold = 0
        valid_loc_count = 0
        
        for recording in test_recs:
            signal = np.load(recording.path)
            result = predict_signal(
                model,
                signal,
                CLASSES,
                locations=locations,
                window_size=4096,
                step=1024,
                aggregation=args.aggregation,
                top_fraction=args.top_fraction,
            )

            true_dev_idx = CLASSES.index(recording.class_name)
            pred_dev_idx = CLASSES.index(str(result["prediction"]))
            dev_correct = int(true_dev_idx == pred_dev_idx)
            correct_dev_fold += dev_correct

            # Location handling
            scenario = recording.scenario
            if scenario in locations:
                true_loc_idx = locations.index(scenario)
                pred_loc_idx = locations.index(str(result["location"]))
                loc_correct = int(true_loc_idx == pred_loc_idx)
                correct_loc_fold += loc_correct
                valid_loc_count += 1
                
                y_loc_true_all.append(true_loc_idx)
                y_loc_pred_all.append(pred_loc_idx)
            else:
                loc_correct = 0

            y_dev_true_all.append(true_dev_idx)
            y_dev_pred_all.append(pred_dev_idx)

            all_predictions_rows.append(
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
                    "split": f"cv_fold_{fold_name}",
                    "device_correct": dev_correct,
                    "location_correct": loc_correct,
                    "windows": int(result["windows"]),
                }
            )

        fold_dev_acc = correct_dev_fold / len(test_recs)
        fold_dev_accuracies.append(fold_dev_acc)
        
        if valid_loc_count > 0:
            fold_loc_acc = correct_loc_fold / valid_loc_count
            fold_loc_accuracies.append(fold_loc_acc)
            print(f"Fold accuracy: device={fold_dev_acc:.4f} ({correct_dev_fold}/{len(test_recs)} correct), location={fold_loc_acc:.4f} ({correct_loc_fold}/{valid_loc_count} correct)")
        else:
            print(f"Fold accuracy: device={fold_dev_acc:.4f} ({correct_dev_fold}/{len(test_recs)} correct)")

    # Overall Summary
    fold_dev_acc_mean = float(np.mean(fold_dev_accuracies))
    fold_dev_acc_std = float(np.std(fold_dev_accuracies))
    
    fold_loc_acc_mean = float(np.mean(fold_loc_accuracies)) if fold_loc_accuracies else 0.0
    fold_loc_acc_std = float(np.std(fold_loc_accuracies)) if fold_loc_accuracies else 0.0

    y_dev_true_array = np.asarray(y_dev_true_all, dtype=np.int64)
    y_dev_pred_array = np.asarray(y_dev_pred_all, dtype=np.int64)
    total_dev_files = len(y_dev_true_array)
    total_dev_correct = int(np.sum(y_dev_true_array == y_dev_pred_array))
    pooled_dev_accuracy = float(total_dev_correct / total_dev_files) if total_dev_files else 0.0
    ci_dev_low, ci_dev_high = wilson_confidence_interval(total_dev_correct, total_dev_files)

    y_loc_true_array = np.asarray(y_loc_true_all, dtype=np.int64)
    y_loc_pred_array = np.asarray(y_loc_pred_all, dtype=np.int64)
    total_loc_files = len(y_loc_true_array)
    total_loc_correct = int(np.sum(y_loc_true_array == y_loc_pred_array))
    pooled_loc_accuracy = float(total_loc_correct / total_loc_files) if total_loc_files else 0.0
    ci_loc_low, ci_loc_high = wilson_confidence_interval(total_loc_correct, total_loc_files)

    print("\n==================================================")
    print(f"CROSS-VALIDATION SUMMARY (Mode: {args.mode.upper()})")
    print(f"Pooling architecture: {args.pooling}")
    print(f"Folds run: {len(folds)}")
    print(f"Fold Device Accuracies: {[f'{acc:.4f}' for acc in fold_dev_accuracies]}")
    print(f"Fold Device Accuracy (Mean ± Std): {fold_dev_acc_mean:.4f} ± {fold_dev_acc_std:.4f}")
    print(f"Pooled File Device Accuracy: {pooled_dev_accuracy:.4f} ({total_dev_correct}/{total_dev_files} correct)")
    print(f"Pooled 95% Wilson Score CI (Device): [{ci_dev_low:.4f}, {ci_dev_high:.4f}]")
    print("--------------------------------------------------")
    print(f"Fold Location Accuracies: {[f'{acc:.4f}' for acc in fold_loc_accuracies]}")
    print(f"Fold Location Accuracy (Mean ± Std): {fold_loc_acc_mean:.4f} ± {fold_loc_acc_std:.4f}")
    print(f"Pooled File Location Accuracy: {pooled_loc_accuracy:.4f} ({total_loc_correct}/{total_loc_files} correct)")
    print(f"Pooled 95% Wilson Score CI (Location): [{ci_loc_low:.4f}, {ci_loc_high:.4f}]")
    print("==================================================")

    # Save outputs
    device_report = write_classification_report(
        y_dev_true_array,
        y_dev_pred_array,
        CLASSES,
        args.output_dir / f"{args.mode}_{args.pooling}_device_classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_dev_true_array,
        y_dev_pred_array,
        CLASSES,
        args.output_dir / f"{args.mode}_{args.pooling}_device_confusion_matrix.png",
    )

    location_report = write_classification_report(
        y_loc_true_array,
        y_loc_pred_array,
        locations,
        args.output_dir / f"{args.mode}_{args.pooling}_location_classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_loc_true_array,
        y_loc_pred_array,
        locations,
        args.output_dir / f"{args.mode}_{args.pooling}_location_confusion_matrix.png",
    )

    write_file_predictions(
        all_predictions_rows,
        args.output_dir / f"{args.mode}_{args.pooling}_predictions.csv",
    )

    metrics = {
        "mode": args.mode,
        "pooling": args.pooling,
        "aggregation": args.aggregation,
        "top_fraction": float(args.top_fraction),
        "epochs": args.epochs,
        "folds": len(folds),
        "mean_fold_device_accuracy": fold_dev_acc_mean,
        "std_fold_device_accuracy": fold_dev_acc_std,
        "pooled_device_accuracy": pooled_dev_accuracy,
        "total_device_files": total_dev_files,
        "total_device_correct": total_dev_correct,
        "pooled_device_ci_95_wilson": {
            "low": ci_dev_low,
            "high": ci_dev_high,
        },
        "mean_fold_location_accuracy": fold_loc_acc_mean,
        "std_fold_location_accuracy": fold_loc_acc_std,
        "pooled_location_accuracy": pooled_loc_accuracy,
        "total_location_files": total_loc_files,
        "total_location_correct": total_loc_correct,
        "pooled_location_ci_95_wilson": {
            "low": ci_loc_low,
            "high": ci_loc_high,
        },
    }

    (args.output_dir / f"{args.mode}_{args.pooling}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved cross-validation metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
