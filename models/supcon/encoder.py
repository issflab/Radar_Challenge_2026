# Wav2Vec2-Large frontend that returns a 4D tensor shaped for CompressionModule:
# (B, K, F, T) where:
#   B = batch, K = number of selected transformer layers,
#   F = feature dim, T = time frames

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Encoder(nn.Module):
    """
    Wraps a pretrained Wav2Vec2 model and exposes stacked hidden states
    as (B, K, F, T) to match the expected input of CompressionModule.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h", freeze_encoder: bool = True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def _forward_frozen(self, waveforms: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(
            waveforms, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        return out.hidden_states

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveforms:      (B, T_samples) float32 mono at 16 kHz.
            attention_mask: (B, T_samples) 1 for real samples, 0 for padding.

        Returns:
            hs_4d: (B, K, F, T) stacked hidden states.
        """
        if attention_mask is None:
            attention_mask = (waveforms != 0.0).long()

        if all(not p.requires_grad for p in self.model.parameters()):
            hidden_states = self._forward_frozen(waveforms, attention_mask)
        else:
            out = self.model(
                waveforms, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
            hidden_states = out.hidden_states

        selected = hidden_states[:]
        hs = torch.stack(selected, dim=0).transpose(0, 1)        # (B, K, T, D)
        hs_4d = hs.permute(0, 1, 3, 2).contiguous()             # (B, K, F=D, T)
        return hs_4d
