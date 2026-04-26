import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import Counter

SR = 16000
N_FFT = 2048
HOP = 512
DURATION = 10
K = 20           # top frequency peaks per file
BIN_HZ = 50      # coarse bin width in Hz for grouping tolerance
OVERLAP = 0.3    # fraction of top-K bins two files must share to be grouped
MIN_FILES = 3    # min group size to bother extracting

DIRS = {
    "false_positives": Path("data/Supcon/false_positives"),
    # "false_negatives": Path("data/Supcon/false_negatives"),
}

FREQ_BINS = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)  # Hz per STFT bin
COARSE_STEP = max(1, int(BIN_HZ / (FREQ_BINS[1] - FREQ_BINS[0])))  # bins per coarse bucket


def get_fingerprint(y):
    """1D frequency fingerprint: top-K peaks in the time-averaged magnitude spectrum."""
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    avg_spectrum = mag.mean(axis=1)                        # collapse time → 1D
    coarse = np.add.reduceat(avg_spectrum,                 # sum into coarse buckets
                             range(0, len(avg_spectrum), COARSE_STEP))
    top_bins = set(np.argsort(coarse)[-K:].tolist())       # top-K coarse freq bins
    return top_bins


def jaccard(a, b):
    return len(a & b) / len(a | b)


def process(tag, audio_dir):
    files = sorted(audio_dir.glob("*.flac"))
    out_dir = audio_dir.parent
    print(f"\n=== {tag}: {len(files)} files ===")

    # Step 1: fingerprint every file
    file_fp = {}
    bin_counter = Counter()
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  fingerprinting {i}/{len(files)}")
        y, _ = librosa.load(f, sr=SR, mono=True, duration=DURATION)
        fp = get_fingerprint(y)
        file_fp[f] = fp
        bin_counter.update(fp)

    # Step 2: greedy grouping — seed on most common bin, expand by overlap
    remaining = list(files)
    groups = []
    for seed_bin, _ in bin_counter.most_common():
        seed_files = [f for f in remaining if seed_bin in file_fp[f]]
        if len(seed_files) < MIN_FILES:
            continue
        # find the consensus fingerprint of the seed group
        consensus = file_fp[seed_files[0]].copy()
        for f in seed_files[1:]:
            consensus &= file_fp[f]
        if not consensus:
            continue
        # expand: any remaining file with sufficient overlap to consensus
        group = [f for f in remaining
                 if len(file_fp[f] & consensus) / K >= OVERLAP]
        if len(group) < MIN_FILES:
            continue
        groups.append(group)
        group_set = set(group)
        remaining = [f for f in remaining if f not in group_set]
        print(f"  Group {len(groups)}: {len(group)} files  "
              f"(consensus bins: {sorted(consensus)[:5]}...)")
        if len(groups) >= 10 or not remaining:
            break

    # Step 3: average waveforms within each group
    print(f"  Extracting {len(groups)} backgrounds...")
    for i, group_files in enumerate(groups):
        avg = np.zeros(SR * DURATION)
        for f in group_files:
            y, _ = librosa.load(f, sr=SR, mono=True, duration=DURATION)
            length = min(len(y), len(avg))
            avg[:length] += y[:length]
        avg /= len(group_files)
        out = out_dir / f"{tag}_bg{i+1}_from{len(group_files)}files.wav"
        sf.write(out, avg, SR)
        print(f"    saved {out.name}")

    if not groups:
        print("  No groups found. Try lowering OVERLAP or MIN_FILES.")


for tag, d in DIRS.items():
    process(tag, d)
