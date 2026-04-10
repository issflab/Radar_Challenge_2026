"""
SupCon two-stage model.

Architecture:
  Stage 1 — Wav2Vec2Encoder + CompressionModule → L2-normalised embedding
  Stage 2 — Linear or MLP binary classifier → scalar logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Wav2Vec2Encoder
from .compression_module import CompressionModule
from ..base import BaseModel


# -------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------
def _safe_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_state_dict_flexible(model: nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


# -------------------------------------------------
# Stage 1
# -------------------------------------------------
class _Stage1Backbone(nn.Module):
    def __init__(self, ckpt_path: str, model_name: str, device: torch.device):
        super().__init__()
        self.encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)

        ckpt = _safe_load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        input_dim = cfg.get("INPUT_DIM", 1024)
        hidden_dim = cfg.get("HIDDEN_DIM", 256)
        dropout = cfg.get("DROPOUT", 0.1)

        if "encoder_state_dict" in ckpt:
            _load_state_dict_flexible(self.encoder, ckpt["encoder_state_dict"])

        self.head = CompressionModule(
            input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout
        ).to(device)
        _load_state_dict_flexible(self.head, ckpt["compression_state_dict"])

        self.encoder.eval()
        self.head.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs_4d = self.encoder(waveforms, attention_mask=attention_mask)
        seq = self.head(hs_4d)
        z = seq.mean(dim=-1)
        return F.normalize(z, p=2, dim=1)


# -------------------------------------------------
# Stage 2
# -------------------------------------------------
class _LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_stage2(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = _safe_load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    head_type = cfg.get("HEAD_TYPE", "linear")
    in_dim = cfg.get("IN_DIM", 256)
    hidden_dim = cfg.get("HIDDEN_DIM", 128)
    dropout = cfg.get("DROPOUT", 0.2)

    if head_type == "linear":
        clf = _LinearHead(in_dim=in_dim).to(device)
    elif head_type == "mlp":
        clf = _MLPHead(in_dim=in_dim, hidden=hidden_dim, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown HEAD_TYPE: {head_type}")

    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf


# -------------------------------------------------
# Public adapter
# -------------------------------------------------
class SupConModel(BaseModel):
    """
    Two-stage SupCon model: Wav2Vec2 backbone (stage 1) + binary head (stage 2).
    """

    def __init__(self, stage1: nn.Module, stage2: nn.Module, device: torch.device):
        self.stage1 = stage1
        self.stage2 = stage2
        self.device = device

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> "SupConModel":
        """
        Expected config keys:
            stage1_ckpt (str) — path to stage-1 .pt checkpoint
            stage2_ckpt (str) — path to stage-2 .pt checkpoint
            model_name  (str) — HuggingFace Wav2Vec2 model id
        """
        stage1 = _Stage1Backbone(
            ckpt_path=config["stage1_ckpt"],
            model_name=config["model_name"],
            device=device,
        )
        stage2 = _load_stage2(ckpt_path=config["stage2_ckpt"], device=device)
        return cls(stage1, stage2, device)

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> float:
        """
        Args:
            waveform: 1-D float tensor (160 000 samples = 10 s at 16 kHz).

        Returns:
            Raw logit — higher means more bonafide.
        """
        x = waveform.unsqueeze(0).to(self.device)        # (1, T)
        attn = torch.ones_like(x, dtype=torch.long)      # (1, T)
        emb = self.stage1(x, attn)                        # (1, hidden_dim)
        return self.stage2(emb).item()
