"""CNN--BiGRU frequency-channel mask model."""

import torch
import torch.nn as nn


N_CHANNELS = 4096
N_TIME_BINS = 4096
VALID_FIRST = 164
VALID_LAST = 3932  # exclusive
DEFAULT_THRESHOLD = 0.38127


class TimeCNN(nn.Module):
    """Encode every frequency channel independently along the time axis."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, out_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nchan, ntime = x.shape
        x = x.reshape(batch * nchan, 1, ntime)
        x = self.net(x).squeeze(-1)
        return x.reshape(batch, nchan, -1).permute(0, 2, 1)


class _TimeEncoder(nn.Module):
    """Compatibility wrapper that preserves keys in the released checkpoint."""

    def __init__(self):
        super().__init__()
        self.encoder = TimeCNN(out_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNNBiGRU(nn.Module):
    """Predict one RFI logit for each frequency channel.

    Input shape: ``[batch, frequency, time]``.
    Output shape: ``[batch, frequency]``.
    """

    def __init__(self):
        super().__init__()
        self.time = _TimeEncoder()
        self.in_proj = nn.Linear(64, 96)
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time(x).permute(0, 2, 1)
        x = self.in_proj(x)
        x, _ = self.gru(x)
        return self.out(x).squeeze(-1)


class ScoreModel(nn.Module):
    """ONNX export wrapper returning bad-channel probabilities."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))
