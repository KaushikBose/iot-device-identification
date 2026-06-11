"""Shared project configuration."""

from __future__ import annotations

WINDOW_SIZE = 4096
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

CLASSES = [
    "dooralarm",
    "lora",
    "microphone",
    "mbus",
    "sigfox",
    "miwi",
]

DATA_FILES = {
    "dooralarm": "dooralarm.npy",
    "lora": "lora.npy",
    "microphone": "microphone.npy",
    "mbus": "mbus.npy",
    "sigfox": "sigfox.npy",
    "miwi": "miwi.npy",
}

TEST_DATA_FILES = {
    class_name: f"{class_name}_test.npy" for class_name in CLASSES
}

SPECTROGRAM_CONFIG = {
    "window": "hann",
    "nperseg": 256,
    "noverlap": 192,
    "nfft": 512,
    "scaling": "spectrum",
}
