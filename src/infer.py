"""Run inference on a raw signal .npy file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from aggregation import AGGREGATION_MODES, aggregate_probabilities
from config import CLASSES, LOCATIONS, WINDOW_SIZE
from model import StopGradientLayer  # noqa: F401 - registers custom layer for load_model
from preprocessing import sliding_windows, windows_to_model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_iot_classifier.h5"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
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


def load_classes(metadata_path: Path) -> list[str]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("classes", CLASSES)
    return CLASSES


def load_locations(metadata_path: Path) -> list[str]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("locations", LOCATIONS)
    return LOCATIONS


def predict_signal(
    model: tf.keras.Model,
    signal: np.ndarray,
    classes: list[str],
    locations: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    step: int = 1024,
    aggregation: str = "mean",
    top_fraction: float = 0.25,
) -> dict[str, object]:
    locations = locations or LOCATIONS
    windows = sliding_windows(signal, window_size=window_size, step=step)
    if len(windows) == 0:
        raise ValueError(
            f"Input signal has {len(signal)} samples, fewer than window_size={window_size}"
        )

    specs = windows_to_model_input(windows)
    device_probs, location_probs = model.predict(specs, verbose=0)
    avg_device_probs = aggregate_probabilities(
        device_probs,
        mode=aggregation,
        top_fraction=top_fraction,
    )
    avg_location_probs = aggregate_probabilities(
        location_probs,
        mode=aggregation,
        top_fraction=top_fraction,
    )
    
    final_device_idx = int(np.argmax(avg_device_probs))
    final_location_idx = int(np.argmax(avg_location_probs))

    return {
        "prediction": classes[final_device_idx],
        "location": locations[final_location_idx],
        "device_confidence": float(avg_device_probs[final_device_idx]),
        "location_confidence": float(avg_location_probs[final_location_idx]),
        "windows": int(len(windows)),
        "aggregation": aggregation,
        "top_fraction": float(top_fraction),
        "device_probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(classes, avg_device_probs)
        },
        "location_probabilities": {
            loc_name: float(probability)
            for loc_name, probability in zip(locations, avg_location_probs)
        },
    }


def main() -> None:
    args = parse_args()
    classes = load_classes(args.metadata)
    locations = load_locations(args.metadata)
    model = tf.keras.models.load_model(str(args.model))
    signal = np.load(args.input)

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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
