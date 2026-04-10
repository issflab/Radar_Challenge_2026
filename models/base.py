from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    """
    Abstract base for all evaluation models.

    Subclasses must implement:
        from_config(config, device) -> instance
        score(waveform)            -> float
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, device: torch.device) -> "BaseModel":
        """Instantiate model from a config dict (model_params section of config.yaml)."""

    @abstractmethod
    def score(self, waveform: torch.Tensor) -> float:
        """
        Score a single utterance.

        Args:
            waveform: 1-D float tensor of exactly TARGET_LEN samples (10 s at 16 kHz).

        Returns:
            Scalar float — higher means more bonafide for models where
            higher_score_means_bonafide=True, lower otherwise.
            The pipeline writes this value directly to the score file.
        """
