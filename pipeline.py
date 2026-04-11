"""
Dataset evaluation pipeline.

Usage:
    python pipeline.py --config config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

from models import MODEL_REGISTRY
from utils.audio import load_fixed_len
from utils.protocol import build_audio_path, load_protocol


def parse_args():
    parser = argparse.ArgumentParser(description="Audio deepfake evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--protocol-file", default=None,
                        help="Override protocol_file from config")
    parser.add_argument("--audio-root", default=None,
                        help="Override audio_root from config (used when path_col=-1)")
    parser.add_argument("--output", default=None,
                        help="Override output_score_file from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model(cfg: dict, device: torch.device):
    model_type = cfg["model_type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type].from_config(cfg["model_params"], device)


def run_eval(cfg: dict, model, device: torch.device) -> None:
    proto_cfg = cfg["protocol"]
    labels, utt_ids, audio_paths = load_protocol(
        path=cfg["protocol_file"],
        key_col=proto_cfg["key_col"],
        label_col=proto_cfg.get("label_col", -1),
        delimiter=proto_cfg.get("delimiter", None),
        path_col=proto_cfg.get("path_col", -1),
    )

    # Audio path resolution: prefer absolute paths from protocol, fall back to root+subdir.
    use_absolute_paths = bool(audio_paths)
    audio_root = cfg.get("audio_root", "")
    audio_subdir = cfg.get("audio_subdir", "")
    extension = cfg.get("audio_extension", ".flac")
    sample_rate = cfg.get("sample_rate", 16_000)

    # Score convention: RADAR wants fake score (higher = more fake).
    # If the model outputs bonafide logits, negate them.
    higher_is_bonafide = cfg["model_params"].get("higher_score_means_bonafide", True)

    output_path = cfg["output_score_file"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    scores: dict[str, float] = {}

    print(f"Scoring {len(utt_ids)} utterances...")
    for i, utt_id in enumerate(utt_ids, 1):
        if use_absolute_paths:
            audio_path = audio_paths[utt_id]
        else:
            audio_path = build_audio_path(audio_root, audio_subdir, utt_id, extension)

        if not os.path.exists(audio_path):
            print(f"  [WARN] Missing audio: {audio_path} — skipping", file=sys.stderr)
            continue

        waveform = load_fixed_len(audio_path, sample_rate=sample_rate)
        raw_score = model.score(waveform)

        # Convert to fake score as required by RADAR challenge.
        fake_score = -raw_score if higher_is_bonafide else raw_score
        scores[utt_id] = fake_score

        if i % 100 == 0 or i == len(utt_ids):
            print(f"  {i}/{len(utt_ids)}")

    _write_score_file(output_path, scores)

    # Compute EER if labels are available.
    if labels:
        _print_eer(scores, labels, utt_ids)


def _write_score_file(output_path: str, scores: dict) -> None:
    """
    Write score.tsv in RADAR challenge format:
      - Tab-separated
      - Header: filename<TAB>score
      - Sorted by utterance ID
      - One line per utterance
    """
    sorted_ids = sorted(scores.keys())
    with open(output_path, "w") as fh:
        fh.write("filename\tscore\n")
        for utt_id in sorted_ids:
            fh.write(f"{utt_id}\t{scores[utt_id]}\n")
    print(f"\nScore file written: {output_path}  ({len(sorted_ids)} utterances)")


def _print_eer(scores: dict, labels: dict, utt_ids: list) -> None:
    from utils.metrics import compute_eer

    bonafide_scores, spoof_scores = [], []
    for utt_id in utt_ids:
        if utt_id not in scores:
            continue
        label = labels.get(utt_id)
        # EER uses raw fake scores: bonafide = low fake score, spoof = high fake score.
        if label == 1:
            bonafide_scores.append(scores[utt_id])
        elif label == 0:
            spoof_scores.append(scores[utt_id])

    if not bonafide_scores or not spoof_scores:
        print("[WARN] Not enough bonafide/spoof samples to compute EER.")
        return

    eer, threshold = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
    print(f"EER: {100 * eer:.2f}%  (threshold: {threshold:.6f})")
    print(f"  Bonafide samples: {len(bonafide_scores)}")
    print(f"  Spoof samples:    {len(spoof_scores)}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.protocol_file is not None:
        cfg["protocol_file"] = args.protocol_file
    if args.audio_root is not None:
        cfg["audio_root"] = args.audio_root
    if args.output is not None:
        cfg["output_score_file"] = args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {cfg['model_type']}")

    print("Loading model...")
    model = get_model(cfg, device)

    run_eval(cfg, model, device)


if __name__ == "__main__":
    main()
