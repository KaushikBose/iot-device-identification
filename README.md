IoT Device Identification from RF Signals

Why I looked into this: As IoT devices proliferate on networks, telecom operators and security teams need to know what's connected — often without cooperation from the device itself. This project explores whether RF signal characteristics alone can reliably identify device type, which is directly relevant to network visibility and security products (e.g. rogue device detection, network access control).

The problem

Traditional device fingerprinting relies on protocol-level metadata (MAC address, DHCP fingerprints), which can be spoofed or unavailable. This project tests a physical-layer approach: can a device be identified purely from the RF spectrogram of its transmissions?

Approach


Convert raw RF signal captures into spectrograms (time-frequency representations)
Train a CNN-based classifier to distinguish device types from spectrogram images
Use a sliding-window ensemble at inference time — classifying multiple overlapping signal windows and aggregating predictions — to improve robustness over a single snapshot


(Add 2-3 sentences here on the dataset used, number of device classes, and your actual accuracy/results once you re-run or extend the notebook.)

What I took away from this (product lens)


Physical-layer identification is a promising complement to protocol-based fingerprinting, but needs enough labeled RF data per device class to generalize — a real constraint for any product built on this approach
Sliding-window ensembling trades inference latency for accuracy — a classic product tradeoff between "detect fast" vs. "detect confidently," which matters for real-time network security use cases

Tech

Python, CNN (deep learning), signal processing (spectrograms)

Further Improvement as per IITD Professors (Going on)

This is a structurally sound pipeline that smartly utilizes a Stop-Gradient layer to prevent location gradients from corrupting device features in the dual-head CNN. Enforcing recording-level data splits to prevent window-level leakage demonstrates strong methodological maturity. However, the reported 100% file-level device accuracy is statistically thin due to the exceptionally small test set of just 18 recordings. Furthermore, the model's 50% accuracy on "another room" predictions highlights a key limitation: discarding phase information in favor of magnitude-only spectrograms washes out the multipath delay signatures needed to resolve environmental ambiguity.

# IoT RF Device Classification

This project classifies recorded RF signals from six IoT device/protocol
classes with a spectrogram-based convolutional neural network (CNN):
`dooralarm`, `lora`, `microphone`, `mbus`, `sigfox`, and `miwi`.

The implemented pipeline converts one-dimensional `.npy` signals into
normalized `257 x 61 x 1` spectrogram inputs, trains a six-class CNN, and
supports both window-level evaluation and whole-file inference using averaged
sliding-window probabilities.

## Documentation

- [Documentation index](docs/index.md)
- [Architecture and diagrams](docs/architecture.md)
- [Methodology and recorded results](docs/methodology.md)
- [Code review findings](docs/code_review.md)
- [Presentation outline](docs/presentation_outline.md)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Training, preprocessing, CNN definition, inference, and evaluation code |
| `data/recordings/` | Recommended multi-capture signals organized by device/scenario |
| `data/recordings_manifest.example.csv` | Template for recording-level train/validation/test assignment |
| `data/raw/`, `data/test/` | Supported legacy one-file-per-class layout |
| `models/` | Saved model binaries when present, metadata, training plots, and held-out report |
| `results/evaluation/` | Window-level evaluation reports |
| `results/test/` | File-level external test reports and probabilities |
| `docs/` | Project documentation and generated architecture diagrams |
| `tests/` | Unit tests; see the documented review findings before relying on them |

Large `.npy` signal files and `.h5`/`.keras` model binaries are excluded from
Git by `.gitignore`.

## Setup

Create a Python 3.10 environment and install the required packages. The
project includes `.python-version` for `pyenv`; Python 3.14 is not supported by
TensorFlow for this project.

```bash
pyenv install -s 3.10.14
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Multi-Recording Dataset

For multiple environments or captures of the same device, use a manifest.
Each row represents one independent recording and assigns the entire recording
to one split:

```csv
path,class,scenario,capture_id,split
data/recordings/microphone/room_capture01.npy,microphone,room,01,train
data/recordings/microphone/background_capture01.npy,microphone,background,01,validation
data/recordings/microphone/upstairs_capture01.npy,microphone,upstairs,01,test
```

Start from `data/recordings_manifest.example.csv` and add every independent
recording. `train` and `validation` must contain every class being trained. A
`test` split may contain all classes or a focused held-out scenario such as
only `microphone` upstairs recordings. Paths are interpreted from the project
working directory. A signal file path may appear only once in the manifest.

## Commands

Train using recording-level splits from a scenario/capture manifest:

```bash
python src/train.py \
  --manifest data/recordings_manifest.csv \
  --pooling avgmax \
  --max-windows-per-class 2000
```

This prevents windows from the same capture appearing in both training and
testing. It saves model files, metadata, curves, and held-out reports in the
chosen output directory. For many long recordings,
`--max-windows-per-class` bounds spectrogram memory and samples across all
captures of a class before conversion.

Run file-level testing of held-out recordings with per-scenario metrics:

```bash
python src/test.py \
  --manifest data/recordings_manifest.csv \
  --manifest-split test \
  --model models/best_iot_classifier.h5 \
  --metadata models/metadata.json \
  --output-dir results
```

Evaluate held-out windows when a window-level diagnostic is needed:

```bash
python src/evaluate.py \
  --manifest data/recordings_manifest.csv \
  --manifest-split test \
  --evaluation-role external-window-evaluation \
  --model models/best_iot_classifier.h5 \
  --metadata models/metadata.json
```

The original one-file-per-class workflow remains supported:

```bash
python src/train.py --data-dir data/raw
python src/test.py --data-dir data/test
```

Classify one raw signal file:

```bash
python src/predict.py --input data/test/miwi_test.npy
```

Create sample spectrogram visualizations:

```bash
python src/visualize.py
```

Regenerate the documentation diagrams:

```bash
python docs/generate_diagrams.py
```

## Recorded Outputs

The checked-in metadata reports `0.9454` test accuracy for the random held-out
window split created during training. `results/evaluation/` reports `0.9900`
accuracy on `data/raw`, which is the same source directory used to build the
training set and should not be presented as independent generalization.
`results/test/` reports six correct external file predictions, one file per
class, using ensemble inference; its metrics include a file-level 95% Wilson
confidence interval to expose the small sample size.

Code review findings and their implemented remediations are tracked in
[docs/code_review.md](docs/code_review.md).
