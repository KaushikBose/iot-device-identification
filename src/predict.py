"""Run a quick prediction on a single test .npy signal file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from aggregation import AGGREGATION_MODES
from config import CLASSES, WINDOW_SIZE
from infer import load_classes, load_locations, predict_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_iot_classifier.h5"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/metadata.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/test/miwi_test.npy"),
        help="Path to the .npy signal file to classify.",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_MODES,
        default="mean",
        help="How to combine per-window probabilities into a whole-file prediction.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction of most confident windows used by top_confidence_mean aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"Model not found at {args.model}. Please train first.")
        return

    if not args.input.exists():
        print(f"Test file not found at {args.input}. Please provide a valid .npy file.")
        return

    classes = load_classes(args.metadata)
    locations = load_locations(args.metadata)

    print("Loading model...")
    model = tf.keras.models.load_model(str(args.model))

    print(f"Loading test data from {args.input}...")
    signal = np.load(args.input)

    print("Running inference...")
    result = predict_signal(
        model,
        signal,
        classes,
        locations=locations,
        window_size=args.window_size,
        step=args.step,
        aggregation=args.aggregation,
        top_fraction=args.top_fraction,
    )

    device = result["prediction"]
    location = result["location"]
    device_conf = result["device_confidence"]
    location_conf = result["location_confidence"]
    n_windows = result["windows"]

    print(f"\n--- Ensemble Results ({n_windows} windows) ---")
    print(f"Final Consolidated Device Prediction: {device}")
    print(f"Device Confidence Level: {device_conf * 100:.2f}%")
    print(f"Final Consolidated Location Prediction: {location}")
    print(f"Location Confidence Level: {location_conf * 100:.2f}%")

    print("\nPer-device probabilities:")
    for class_name, prob in result["device_probabilities"].items():
        print(f"  {class_name:>12s}: {prob * 100:.2f}%")

    print("\nPer-location probabilities:")
    for loc_name, prob in result["location_probabilities"].items():
        print(f"  {loc_name:>12s}: {prob * 100:.2f}%")

    print("\nConfidence is an averaged model score across windows, not an accuracy estimate.")


if __name__ == "__main__":
    main()
