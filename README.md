# RADAR Challenge 2026 Ensemble Evaluation

This repository runs inference for one or more spoofing detection models on RADAR evaluation audio and generates a score file from the averaged model outputs.

The main use case is ensemble evaluation:
- load one or more checkpoints from `config.yaml`
- run all models on the same input utterances
- average their output scores
- save a final score file for downstream evaluation or submission

The single-model case is handled by the same pipeline with a `model_paths` list containing one checkpoint.

## Repository Contents

- `test.py`: main evaluation script for loading checkpoints, running inference, and writing the score file
- `config.yaml`: evaluation configuration
- `data_utils_SSL.py`: protocol parsing and RADAR dataset loading
- `evaluation.py`: metric utilities
- `generate_protocol.py`: utility for creating a protocol file from a directory of audio files
- `model.py`: model definition
- `feature_extraction.py`: feature extraction utilities used by the model

## Configuration

All runtime configuration is read from `config.yaml`.

### Top-Level Parameters

- `model_paths`
  Paths to model checkpoints.
  Use one path for single-model evaluation or multiple paths for ensemble evaluation.

- `score_dir`
  Directory where generated score files will be written.

- `eval_output`
  Name of the output score file if you want to keep a descriptive filename in the config.
  The current `test.py` script writes scores to `<dataset>_eval_score.txt` inside `score_dir`.

- `cuda_device`
  CUDA device string used for inference, for example `cuda:0`.
  If CUDA is not available, evaluation falls back to CPU.

### `data_config`

- `dataset`
  Dataset name used to name the output score file.

- `data_dir`
  Directory containing the evaluation audio files.
  In the current RADAR dataset loader, the code expects audio files as:
  `<data_dir>/<file_id>.flac`

- `protocol_path`
  Directory containing the protocol file.

- `protocol_filename`
  Name of the protocol file inside `protocol_path`.

- `protocol_delimiter`
  Delimiter used in the protocol file, for example `","` or `" "`.

- `protocol_file_id_column`
  Zero-based column index for the file identifier in the protocol file.
  This is the ID used to locate each audio file.

- `protocol_label_column`
  Zero-based column index for the label in the protocol file.
  This is mainly relevant when protocol parsing is used with labels enabled.

- `bonfide_label`
  String value used in the protocol to represent bona fide speech.

- `num_samples`
  Number of waveform samples used per utterance after trimming or padding.

- `sample_rate`
  Audio sample rate expected by the pipeline.


### Example `config.yaml`

```yaml
model_paths:
  - /path/to/model_a.pth
  - /path/to/model_b.pth
score_dir: Score_Files
eval_output: radar_eval_scores.txt
cuda_device: cuda:0

data_config:
  dataset: radar_challenge_dev
  data_dir: /data/radar_challenge_data/RADAR2026-dev/flac
  protocol_path: /data/radar_challenge_data/RADAR2026-dev
  protocol_filename: dev_protocol.txt
  protocol_delimiter: ","
  protocol_file_id_column: 0
  protocol_label_column: 1
  bonfide_label: bona-fide
  num_samples: 192000
  sample_rate: 16000
  duration: 4.0375
  eval_duration: 4.0375
  num_workers: 4
  augment: "False"
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Radar_Challenge_2026
```

### 2. Create a virtual environment

```bash
conda create -n radar_eval python=3.10
conda activate radar_eval
```

### 3. Install dependencies

If `requirements.txt` is complete for your environment:

```bash
pip install -r requirements.txt
```

If you need the core packages manually, install the versions appropriate for your CUDA and PyTorch setup:

```bash
pip install torch torchaudio pandas pyyaml tqdm librosa numpy
```

## Preparing a Protocol File

If you need to generate a protocol file from a directory of audio files, use `generate_protocol.py`.

Example:

```bash
python3 generate_protocol.py \
  --input_dir /data/radar_challenge_data/RADAR2026-dev \
  --protocol_name dev_protocol.txt \
  --extension .flac \
  --delimiter ","
```

## Running Evaluation

Update `config.yaml` with:
- the model checkpoint paths in `model_paths`
- the correct RADAR audio directory in `data_config.data_dir`
- the protocol location and format in `data_config`
- the desired output directory in `score_dir`

Then run:

```bash
python3 test.py --config config.yaml
```


## Score File Output

The evaluation script writes the score file to:

```text
<score_dir>/<dataset>_eval_score.txt
```

For the sample config above, that would be:

```text
Score_Files/radar_challenge_dev_eval_score.txt
```

The current output format is:

```text
<utt_id> <score>
```

where:
- `utt_id` is the protocol file ID
- `score` is the ensemble-averaged model score

## Ensemble Behavior

- If `model_paths` contains one checkpoint, the script behaves like a normal single-model evaluator.
- If `model_paths` contains multiple checkpoints, all models are loaded and evaluated on each batch.
- The model outputs are averaged before writing the final score file.
- All ensemble members are expected to use the same architecture and output shape.
