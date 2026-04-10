import torch
import torch.nn as nn


class CompressionModule(nn.Module):
    """
    Fuses and processes encoder features to produce a final embedding.
    Input:  (B, K, F, T)
    Output: (B, hidden_dim, T)
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_head = nn.Dropout(p=dropout_rate)
        self.activation_head = nn.LeakyReLU()
        self.mlp3 = nn.Linear(input_dim, hidden_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(encoder_output, dim=1)           # (B, F, T)
        x = self.dropout_head(pooled)
        x = self.activation_head(x)
        return self.mlp3(x.transpose(1, 2)).transpose(1, 2)  # (B, hidden_dim, T)
