"""EER computation — self-contained, no external repo dependencies."""

import numpy as np


def _compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple:
    """
    Returns (EER, threshold) where EER is in [0, 1].

    Args:
        target_scores:    Scores for bonafide utterances.
        nontarget_scores: Scores for spoof utterances.
    """
    frr, far, thresholds = _compute_det_curve(target_scores, nontarget_scores)
    min_index = np.argmin(np.abs(frr - far))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])
