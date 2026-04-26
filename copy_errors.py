"""Copy FP and FN audio files at the EER threshold to data/Supcon/."""

import sys, shutil, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics import compute_eer

SCORE_TSV  = Path(__file__).parent / "scores/RADAR2026-dev/score.tsv"
LABEL_FILE = Path(__file__).parent / "label_RADAR2026-dev.txt"
AUDIO_DIR  = Path("/data/radar_challenge_data/RADAR2026-dev/flac")
OUT_FP     = Path(__file__).parent / "data/Supcon/false_positives"   # spoof predicted bonafide
OUT_FN     = Path(__file__).parent / "data/Supcon/false_negatives"   # bonafide predicted spoof
OUT_FP.mkdir(parents=True, exist_ok=True)
OUT_FN.mkdir(parents=True, exist_ok=True)

# Load labels
labels = {}
with open(LABEL_FILE) as f:
    for line in f:
        p = line.split()
        labels[p[0]] = p[2]

# Load scores
scores = {}
with open(SCORE_TSV) as f:
    for line in f:
        p = line.split()
        if p[0] == "filename":
            continue
        scores[p[0]] = float(p[1])

# Compute EER threshold (fake score space)
bonafide = [scores[u] for u in scores if labels.get(u) == "bonafide"]
spoof    = [scores[u] for u in scores if labels.get(u) == "spoof"]
_, neg_thr = compute_eer(-np.array(bonafide), -np.array(spoof))
threshold = -neg_thr
print(f"EER threshold (fake score): {threshold:.4f}")

# Classify and copy
fp, fn = [], []
for utt_id, score in scores.items():
    label = labels.get(utt_id)
    if label is None:
        continue
    predicted_spoof = score > threshold
    if label == "spoof" and not predicted_spoof:    # FP: spoof → predicted bonafide
        fp.append(utt_id)
    elif label == "bonafide" and predicted_spoof:   # FN: bonafide → predicted spoof
        fn.append(utt_id)

print(f"False positives (spoof→bonafide): {len(fp)}")
print(f"False negatives (bonafide→spoof): {len(fn)}")

for utt_id in fp:
    src = AUDIO_DIR / f"{utt_id}.flac"
    if src.exists():
        shutil.copy2(src, OUT_FP / src.name)

for utt_id in fn:
    src = AUDIO_DIR / f"{utt_id}.flac"
    if src.exists():
        shutil.copy2(src, OUT_FN / src.name)

print(f"Copied to {OUT_FP.parent}/")
print(f"  false_positives/: {len(list(OUT_FP.iterdir()))} files")
print(f"  false_negatives/: {len(list(OUT_FN.iterdir()))} files")
