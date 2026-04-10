# RADAR Challenge 2026 — Evaluation Pipeline

A config-driven pipeline for running audio deepfake detection models on RADAR Challenge datasets and generating submission-ready score files.

Supports single-model evaluation and ensemble averaging out of the box. Adding a new model requires no changes to the pipeline code — just drop a folder into `models/`.

---

## Repository Layout

```
Radar_Challenge_2026/
├── pipeline.py              # Entry point
├── config.yaml              # All runtime configuration
├── requirements.txt
├── checkpoints/             # Place model checkpoint files here
│   └── supcon/              # One subfolder per model
├── models/
│   ├── __init__.py          # Model registry (model_type → class)
│   ├── base.py              # Abstract BaseModel interface
│   └── supcon/              # Self-contained model package
│       ├── __init__.py
│       ├── model.py         # SupConModel + Stage 1 / Stage 2
│       ├── encoder.py       # Wav2Vec2Encoder
│       └── compression_module.py
├── utils/
│   ├── audio.py             # Fixed-length audio loading (pad / trim to 10 s)
│   ├── protocol.py          # Protocol file parsing
│   └── metrics.py           # EER computation
└── scores/                  # Generated score files land here
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your checkpoint files under `checkpoints/<model_name>/`.

---

## Configuration (`config.yaml`)

| Field | Description |
|-------|-------------|
| `model_type` | Key in the model registry — must match a key in `models/__init__.py` |
| `model_params` | Model-specific settings (checkpoint paths, HuggingFace model id, etc.) |
| `protocol_file` | Path to the protocol CSV |
| `protocol.key_col` | Column index of the utterance ID |
| `protocol.path_col` | Column index of the absolute audio path (`-1` to use `audio_root` instead) |
| `protocol.label_col` | Column index of the label (`-1` if no labels — EER will be skipped) |
| `protocol.delimiter` | Column separator (`,` for RADAR CSV format) |
| `output_score_file` | Where to write the score file |
| `sample_rate` | Audio sample rate (default `16000`) |

### Protocol format (RADAR 2026)

```
RADAR2026-DEV000001,/abs/path/to/RADAR2026-DEV000001.flac
RADAR2026-DEV000002,/abs/path/to/RADAR2026-DEV000002.flac
```

### Audio from a root directory (ASVspoof-style)

If your protocol does not include audio paths, set `protocol.path_col: -1` and configure:

```yaml
audio_root: /path/to/dataset/
audio_subdir: flac        # subdirectory under audio_root; "" for flat layout
audio_extension: .flac
```

---

## Running Evaluation

```bash
python pipeline.py --config config.yaml
```

The pipeline will:
1. Parse the protocol file
2. Load the model specified by `model_type`
3. Load each audio file, pad or trim to 10 seconds
4. Run inference — one score per utterance
5. Write `score.tsv` (sorted by utterance ID)
6. Print EER if labels are present in the protocol

### Score file format

```
filename	score
RADAR2026-DEV000001	4.594078
RADAR2026-DEV000002	3.183919
...
```

Tab-separated, header line included. Higher score = higher probability of being **fake**. The pipeline automatically converts bonafide logits to fake scores when `higher_score_means_bonafide: true` is set.

### Submission packaging

```bash
cd scores/RADAR2026-dev
zip ../../submission.zip score.tsv
```

---

## Adding a New Model

### 1. Create a self-contained model folder

```
models/
└── my_model/
    ├── __init__.py
    ├── model.py          # Your model class
    └── <any architecture files>
```

### 2. Implement `BaseModel` in `model.py`

```python
from ..base import BaseModel
import torch

class MyModel(BaseModel):

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> "MyModel":
        # config = the model_params section from config.yaml
        # Load checkpoints, build architecture, return an instance
        ...

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> float:
        # waveform: 1-D float32 tensor, 160 000 samples (10 s at 16 kHz)
        # Return a single float — higher means more bonafide
        # (set higher_score_means_bonafide: true in config.yaml)
        ...
```

Export the class from `models/my_model/__init__.py`:

```python
from .model import MyModel
```

### 3. Register the model

In `models/__init__.py`, add one import and one registry entry:

```python
from .my_model import MyModel

MODEL_REGISTRY: dict = {
    "supcon": SupConModel,
    "my_model": MyModel,   # add this line
}
```

### 4. Add a config entry

```yaml
model_type: my_model

model_params:
  # whatever your from_config() expects
  checkpoint: checkpoints/my_model/weights.pt
```

That's it — the rest of the pipeline (audio loading, protocol parsing, score writing, EER) works unchanged.

---

## Score Convention

- The pipeline calls `model.score(waveform)` and gets a raw float.
- If `model_params.higher_score_means_bonafide: true`, the pipeline negates the score before writing (`fake_score = -raw_score`).
- If `model_params.higher_score_means_bonafide: false`, the raw score is written directly as the fake score.

Set this flag to match your model's output convention. The RADAR challenge requires higher score = more fake.
