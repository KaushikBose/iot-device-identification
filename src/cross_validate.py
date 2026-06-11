"""Cross-validation runner for Leave-One-Recording-Out (LORO) and Leave-One-Environment-Out (LOEO)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import BATCH_SIZE, CLASSES, EPOCHS, NUM_CLASSES, SEED
from dataset import build_dataset_from_recordings, load_manifest
from evaluation import (
    save_confusion_matrix_plot,
    write_classification_report,
)
from infer import predict_signal
from model import build_cnn
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
    folds = get_folds(recordings, args.mode)
    print(f"Starting Cross-Validation (Mode: {args.mode.upper()}) with {len(folds)} folds.")
    print(f"Architecture configuration: pooling={args.pooling}, epochs={args.epochs}")

    fold_accuracies: list[float] = []
    all_predictions_rows: list[dict[str, object]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for idx, (fold_name, train_recs, test_recs) in enumerate(folds, start=1):
        print(f"\n--- Fold {idx}/{len(folds)}: {fold_name} ---")
        print(f"Training on {len(train_recs)} recordings. Testing on {len(test_recs)} recordings.")

        # Build training / validation window datasets
        # We extract all windows from train_recs and split them 85% train, 15% val
        x_train_val, y_train_val = build_dataset_from_recordings(
            train_recs,
            classes=CLASSES,
            balance=not args.no_balance,
            seed=args.seed,
            max_windows_per_class=args.max_windows_per_class,
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.15,
            stratify=y_train_val,
            random_state=args.seed,
        )

        y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

        # Build and compile model
        model = build_cnn(
            input_shape=x_train.shape[1:],
            num_classes=NUM_CLASSES,
            learning_rate=args.learning_rate,
            pooling=args.pooling,
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        # Fit model
        model.fit(
            x_train,
            y_train_cat,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val_cat),
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on test recordings at the file (ensemble) level
        correct_fold = 0
        for recording in test_recs:
            signal = np.load(recording.path)
            result = predict_signal(
                model,
                signal,
                CLASSES,
                window_size=4096,
                step=1024,
            )

            true_idx = CLASSES.index(recording.class_name)
            pred_idx = CLASSES.index(str(result["prediction"]))
            correct = int(true_idx == pred_idx)
            correct_fold += correct

            y_true_all.append(true_idx)
            y_pred_all.append(pred_idx)

            all_predictions_rows.append(
                {
                    "file": str(recording.path),
                    "true_label": recording.class_name,
                    "scenario": recording.scenario,
                    "capture_id": recording.capture_id,
                    "split": f"cv_fold_{fold_name}",
                    "predicted_label": result["prediction"],
                    "confidence": f"{float(result['confidence']):.8f}",
                    "windows": int(result["windows"]),
                    "correct": correct,
                }
            )

        fold_acc = correct_fold / len(test_recs)
        fold_accuracies.append(fold_acc)
        print(f"Fold accuracy: {fold_acc:.4f} ({correct_fold}/{len(test_recs)} correct files)")

    # Overall Summary
    fold_acc_mean = float(np.mean(fold_accuracies))
    fold_acc_std = float(np.std(fold_accuracies))

    y_true_array = np.asarray(y_true_all, dtype=np.int64)
    y_pred_array = np.asarray(y_pred_all, dtype=np.int64)
    total_files = len(y_true_array)
    total_correct = int(np.sum(y_true_array == y_pred_array))
    pooled_accuracy = float(total_correct / total_files) if total_files else 0.0

    ci_low, ci_high = wilson_confidence_interval(total_correct, total_files)

    print("\n==================================================")
    print(f"CROSS-VALIDATION SUMMARY (Mode: {args.mode.upper()})")
    print(f"Pooling architecture: {args.pooling}")
    print(f"Folds run: {len(folds)}")
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Fold Accuracy (Mean ± Std): {fold_acc_mean:.4f} ± {fold_acc_std:.4f}")
    print(f"Pooled File Accuracy: {pooled_accuracy:.4f} ({total_correct}/{total_files} correct)")
    print(f"Pooled 95% Wilson Score CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print("==================================================")

    # Save outputs
    report = write_classification_report(
        y_true_array,
        y_pred_array,
        CLASSES,
        args.output_dir / f"{args.mode}_{args.pooling}_classification_report.txt",
    )
    save_confusion_matrix_plot(
        y_true_array,
        y_pred_array,
        CLASSES,
        args.output_dir / f"{args.mode}_{args.pooling}_confusion_matrix.png",
    )
    write_file_predictions(
        all_predictions_rows,
        args.output_dir / f"{args.mode}_{args.pooling}_predictions.csv",
    )

    metrics = {
        "mode": args.mode,
        "pooling": args.pooling,
        "epochs": args.epochs,
        "folds": len(folds),
        "fold_accuracies": fold_accuracies,
        "mean_fold_accuracy": fold_acc_mean,
        "std_fold_accuracy": fold_acc_std,
        "pooled_accuracy": pooled_accuracy,
        "total_files": total_files,
        "total_correct": total_correct,
        "pooled_ci_95_wilson": {
            "low": ci_low,
            "high": ci_high,
        },
    }

    (args.output_dir / f"{args.mode}_{args.pooling}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved cross-validation metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
