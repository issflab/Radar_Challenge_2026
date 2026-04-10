"""Fixed-length audio loading: load, mono, resample, pad or trim to exactly TARGET_LEN samples."""

import numpy as np
import librosa
import torch

TARGET_LEN = 160_000  # 10 seconds at 16 kHz


def load_fixed_len(path: str, sample_rate: int = 16_000, target_len: int = TARGET_LEN) -> torch.Tensor:
    """
    Load an audio file and return exactly target_len samples.

    - Resamples to sample_rate.
    - If shorter than target_len: tiles (repeats) to fill, then truncates.
    - If longer: truncates to first target_len samples.

    Returns:
        torch.Tensor of shape (target_len,), dtype float32.
    """
    y, _ = librosa.load(path, sr=sample_rate, mono=True)

    if len(y) == 0:
        return torch.zeros(target_len, dtype=torch.float32)

    if len(y) < target_len:
        reps = (target_len + len(y) - 1) // len(y)
        y = np.tile(y, reps)

    y = y[:target_len]
    return torch.from_numpy(y).float()
