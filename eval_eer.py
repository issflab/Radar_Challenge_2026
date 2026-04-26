"""Compute EER for score files against the dev label file."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics import compute_eer

LABEL_FILE = Path(__file__).parent / "label_RADAR2026-dev.txt"
SCORE_FILES = [
    Path(__file__).parent / "scores/surya_model/output_score_radar_small.txt",
    Path(__file__).parent / "scores/surya_model/output_score_radar_large.txt",
    Path(__file__).parent / "scores/RADAR2026-dev/score.tsv",
    Path(__file__).parent / "scores/llamapartialspoof/score.tsv",
]

# Load labels: {utt_id (no ext) -> "bonafide"/"spoof"}
labels = {}
with open(LABEL_FILE) as f:
    for line in f:
        parts = line.split()
        labels[parts[0]] = parts[2]  # e.g. RADAR2026-DEV000001 -> bonafide

for score_file in SCORE_FILES:
    bonafide, spoof, skipped = [], [], 0
    with open(score_file) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "filename":  # skip TSV header
                continue
            utt_id = parts[0].replace(".flac", "")
            score = float(parts[1])
            label = labels.get(utt_id)
            if label == "bonafide":
                bonafide.append(score)
            elif label == "spoof":
                spoof.append(score)
            else:
                skipped += 1

    if not bonafide or not spoof:
        print(f"{score_file.name}: no matching labels found (skipped {skipped} utterances)")
        continue

    # Negate: scores are fake scores (higher=spoof), compute_eer expects higher=bonafide
    eer, neg_thr = compute_eer(-np.array(bonafide), -np.array(spoof))
    skip_note = f"  [{skipped} utt not in labels]" if skipped else ""
    print(f"{score_file.name}: EER = {eer*100:.2f}%  (threshold = {-neg_thr:.4f}){skip_note}")
