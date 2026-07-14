"""Dataset utilities shared by training and evaluation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model import N_CHANNELS, N_TIME_BINS


def normalize_intensity(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    median = np.nanmedian(array)
    sigma = 1.4826 * np.nanmedian(np.abs(array - median))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(array)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    array = (array - median) / sigma
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


class BurstMaskDataset(Dataset):
    """Read paired ``*.I.npy`` and ``*.mask.npy`` files on demand.

    The stored mask uses 1 for a retained channel and 0 for a masked channel.
    The training target is inverted so that 1 denotes an RFI-contaminated channel.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.intensity_files = sorted(self.directory.glob("*.I.npy"))
        if not self.intensity_files:
            raise FileNotFoundError(f"No *.I.npy files found in {self.directory}")

        missing = [p for p in self.intensity_files if not self._mask_path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing mask for {missing[0]}")

    @staticmethod
    def _mask_path(intensity_path: Path) -> Path:
        return Path(str(intensity_path).replace(".I.npy", ".mask.npy"))

    def __len__(self) -> int:
        return len(self.intensity_files)

    def __getitem__(self, index: int):
        intensity_path = self.intensity_files[index]
        intensity = normalize_intensity(np.load(intensity_path))
        mask = np.asarray(np.load(self._mask_path(intensity_path)), dtype=np.uint8)

        if intensity.shape != (N_CHANNELS, N_TIME_BINS):
            raise ValueError(f"Unexpected intensity shape {intensity.shape} in {intensity_path}")
        if mask.shape != (N_CHANNELS,):
            raise ValueError(f"Unexpected mask shape {mask.shape} in {intensity_path}")

        target = 1.0 - mask.astype(np.float32)
        return torch.from_numpy(intensity), torch.from_numpy(target), intensity_path.name
