"""
Protocol file parsing — self-contained, no external repo dependencies.

Supported formats (configurable via column indices and delimiter):

  RADAR 2026 (2-col CSV, no labels, absolute paths):
    RADAR2026-DEV000001,/abs/path/to/RADAR2026-DEV000001.flac
    → key_col=0, path_col=1, label_col=-1, delimiter=","

  ASVspoof training (5-col, space-separated):
    speaker_id  utt_id  -  -  label
    → key_col=1, label_col=4, path_col=-1, delimiter=None

  ASVspoof eval (7-col, space-separated):
    -  utt_id  -  -  -  label  phase
    → key_col=1, label_col=5, path_col=-1, delimiter=None
"""

import os
from typing import Optional


def _normalize_delim(d: Optional[str]) -> Optional[str]:
    """
    None / "" / " " → split on any whitespace.
    Any other string  → used as-is in str.split(d).
    """
    if d is None:
        return None
    if isinstance(d, str) and d.strip() in ("", " "):
        return None
    return d


def load_protocol(
    path: str,
    key_col: int,
    label_col: int,
    delimiter: Optional[str],
    path_col: int = -1,
) -> tuple:
    """
    Parse a protocol file.

    Args:
        path:      Path to the protocol text file.
        key_col:   Column index of the utterance ID.
        label_col: Column index of the label ('bonafide'/'spoof').
                   Pass -1 to skip label parsing.
        delimiter: Column separator. None / "" / " " means any whitespace.
        path_col:  Column index of the absolute audio path.
                   Pass -1 when audio paths are built from audio_root instead.

    Returns:
        (labels, utt_ids, audio_paths)
            labels:      dict utt_id -> int (1=bonafide, 0=spoof).
                         Empty dict when label_col == -1.
            utt_ids:     Ordered list of utterance IDs (preserves protocol order).
            audio_paths: dict utt_id -> str absolute path.
                         Empty dict when path_col == -1.
    """
    delim = _normalize_delim(delimiter)
    utt_ids = []
    labels = {}
    audio_paths = {}
    has_label = label_col >= 0
    has_path = path_col >= 0

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delim) if delim is not None else line.split()

            if key_col >= len(parts):
                raise ValueError(f"key_col {key_col} out of range for line: `{line}`")
            if has_label and label_col >= len(parts):
                raise ValueError(f"label_col {label_col} out of range for line: `{line}`")
            if has_path and path_col >= len(parts):
                raise ValueError(f"path_col {path_col} out of range for line: `{line}`")

            utt_id = parts[key_col]
            utt_ids.append(utt_id)

            if has_label:
                labels[utt_id] = 1 if parts[label_col] == "bonafide" else 0
            if has_path:
                audio_paths[utt_id] = parts[path_col]

    return labels, utt_ids, audio_paths


def build_audio_path(audio_root: str, audio_subdir: str, utt_id: str, extension: str) -> str:
    """Construct the full path to an audio file from its utterance ID."""
    if audio_subdir:
        return os.path.join(audio_root, audio_subdir, utt_id + extension)
    return os.path.join(audio_root, utt_id + extension)
