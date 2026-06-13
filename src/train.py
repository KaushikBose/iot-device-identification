"""Train the RF IoT spectrogram classifier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import BATCH_SIZE, CLASSES, EPOCHS, NUM_CLASSES, NUM_LOCATIONS, SEED, LOCATIONS
from dataset import build_dataset, load_manifest, split_dataset
from evaluation import (
    save_confusion_matrix_plot,
    save_training_curves,
    write_classification_report,
)
from model import build_cnn
from preprocessing import class_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV recording manifest with explicit train, validation, and test splits. Overrides --data-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["avg", "max", "avgmax", "flatten"],
        default="avg",
        help="Pooling layer to use before classification head.",
    )
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Optionally sample at most this many windows per class in each split before spectrogram conversion.",
    )
    return parser.parse_args()


def manifest_summary(manifest_path: Path) -> dict[str, object]:
    """Summarize recording and scenario assignments for model metadata."""
    recordings = load_manifest(manifest_path, classes=CLASSES)
    split_counts = Counter(recording.split or "unspecified" for recording in recordings)
    scenario_counts = Counter(recording.scenario for recording in recordings)
    classes_by_split: dict[str, set[str]] = {}
    for recording in recordings:
        split = recording.split or "unspecified"
        classes_by_split.setdefault(split, set()).add(recording.class_name)
    return {
        "manifest": str(manifest_path),
        "recordings_by_split": dict(sorted(split_counts.items())),
        "recordings_by_scenario": dict(sorted(scenario_counts.items())),
        "classes_by_split": {
            split: sorted(class_names)
            for split, class_names in sorted(classes_by_split.items())
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.manifest:
        x_train, y_dev_train, y_loc_train = build_dataset(
            manifest_path=args.manifest,
            split="train",
            classes=CLASSES,
            locations=LOCATIONS,
            balance=not args.no_balance,
            seed=args.seed,
            max_windows_per_class=args.max_windows_per_class,
        )
        x_val, y_dev_val, y_loc_val = build_dataset(
            manifest_path=args.manifest,
            split="validation",
            classes=CLASSES,
            locations=LOCATIONS,
            balance=not args.no_balance,
            seed=args.seed,
            max_windows_per_class=args.max_windows_per_class,
        )
        x_test, y_dev_test, y_loc_test = build_dataset(
            manifest_path=args.manifest,
            split="test",
            classes=CLASSES,
            locations=LOCATIONS,
            balance=not args.no_balance,
            seed=args.seed,
            allow_missing=True,
            max_windows_per_class=args.max_windows_per_class,
        )
        split_strategy = "recording_manifest"
        source_metadata = manifest_summary(args.manifest)
    else:
        x, y_device, y_location = build_dataset(
            args.data_dir,
            classes=CLASSES,
            locations=LOCATIONS,
            balance=not args.no_balance,
            seed=args.seed,
            max_windows_per_class=args.max_windows_per_class,
        )
        print(f"Dataset shape: {x.shape}")
        print(f"Device label distribution: {class_distribution(y_device)}")
        print(f"Location label distribution: {class_distribution(y_location)}")
        (
            x_train, x_val, x_test,
            y_dev_train, y_dev_val, y_dev_test,
            y_loc_train, y_loc_val, y_loc_test,
        ) = split_dataset(
            x,
            y_device,
            y_location,
            test_size=0.10,
            validation_size=0.10,
            seed=args.seed,
        )
        split_strategy = "random_window_split_legacy"
        source_metadata = {"data_dir": str(args.data_dir)}

    print(f"Split strategy: {split_strategy}")
    print(f"Train shape/device distribution: {x_train.shape} / {class_distribution(y_dev_train)}")
    print(f"Train location distribution: {class_distribution(y_loc_train)}")
    print(f"Validation shape/device distribution: {x_val.shape} / {class_distribution(y_dev_val)}")
    print(f"Test shape/device distribution: {x_test.shape} / {class_distribution(y_dev_test)}")

    y_dev_train_cat = tf.keras.utils.to_categorical(y_dev_train, NUM_CLASSES)
    y_dev_val_cat = tf.keras.utils.to_categorical(y_dev_val, NUM_CLASSES)
    y_dev_test_cat = tf.keras.utils.to_categorical(y_dev_test, NUM_CLASSES)

    y_loc_train_cat = tf.keras.utils.to_categorical(y_loc_train, NUM_LOCATIONS)
    y_loc_val_cat = tf.keras.utils.to_categorical(y_loc_val, NUM_LOCATIONS)
    y_loc_test_cat = tf.keras.utils.to_categorical(y_loc_test, NUM_LOCATIONS)

    model = build_cnn(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
        num_locations=NUM_LOCATIONS,
        learning_rate=args.learning_rate,
        pooling=args.pooling,
    )
    model.summary()

    best_model_path = args.output_dir / "best_iot_classifier.h5"
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_device_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_device_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_device_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        {"device": y_dev_train_cat, "location": y_loc_train_cat},
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            x_val,
            {"device": y_dev_val_cat, "location": y_loc_val_cat},
        ),
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = args.output_dir / "final_iot_classifier.h5"
    model.save(str(final_model_path))

    test_metrics = model.evaluate(
        x_test,
        {"device": y_dev_test_cat, "location": y_loc_test_cat},
        verbose=0,
        return_dict=True,
    )

    pred_device_probs, pred_location_probs = model.predict(x_test)
    pred_device_classes = np.argmax(pred_device_probs, axis=1)
    pred_location_classes = np.argmax(pred_location_probs, axis=1)

    device_report_path = args.output_dir / "device_classification_report.txt"
    device_report = write_classification_report(y_dev_test, pred_device_classes, CLASSES, device_report_path)
    save_confusion_matrix_plot(
        y_dev_test,
        pred_device_classes,
        CLASSES,
        args.output_dir / "device_confusion_matrix.png",
    )

    location_report_path = args.output_dir / "location_classification_report.txt"
    location_report = write_classification_report(y_loc_test, pred_location_classes, LOCATIONS, location_report_path)
    save_confusion_matrix_plot(
        y_loc_test,
        pred_location_classes,
        LOCATIONS,
        args.output_dir / "location_confusion_matrix.png",
    )

    save_training_curves(history, args.output_dir / "training_curves.png")

    metadata = {
        "classes": CLASSES,
        "locations": LOCATIONS,
        "split_strategy": split_strategy,
        "source": source_metadata,
        "test_evaluation_role": (
            "held-out-recordings-from-manifest"
            if args.manifest
            else "random-window-holdout-from-training-source-recordings"
        ),
        "test_evaluation_unit": "non_overlapping_window",
        "test_samples": int(len(y_dev_test)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "input_shape": list(x_train.shape[1:]),
        "balanced": not args.no_balance,
        "max_windows_per_class": args.max_windows_per_class,
        "pooling": args.pooling,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nTest device accuracy: {test_metrics.get('device_accuracy', 0.0):.4f}")
    print(f"Test location accuracy: {test_metrics.get('location_accuracy', 0.0):.4f}")
    print("\n--- Device Classification Report ---")
    print(device_report)
    print("\n--- Location Classification Report ---")
    print(location_report)
    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
