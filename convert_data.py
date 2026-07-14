"""Convert manually cleaned PSRCHIVE archives to NumPy training pairs.

Input archives must already be folded and manually inspected with ``pazi``.
Their channel weights are used as the reference masks.
"""

from pathlib import Path

import numpy as np
import psrchive


# =========================
# Settings
# =========================

INPUT_DIR = Path("archives").expanduser()
OUTPUT_DIR = Path("data/converted").expanduser()
ARCHIVE_SUFFIXES = (".dm.pazi", ".ar.dm", ".dm")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
archives = sorted(
    path
    for path in INPUT_DIR.iterdir()
    if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)
)
if not archives:
    raise FileNotFoundError(f"No PSRCHIVE files found in {INPUT_DIR}")

failed = []
converted = 0

for index, archive_path in enumerate(archives, 1):
    intensity_path = OUTPUT_DIR / f"{archive_path.name}.I.npy"
    mask_path = OUTPUT_DIR / f"{archive_path.name}.mask.npy"

    try:
        archive = psrchive.Archive_load(str(archive_path))
        if archive.get_state() != "Stokes":
            archive.convert_state("Stokes")
        archive.remove_baseline()

        data = archive.get_data()       # [subintegration, polarization, channel, bin]
        weights = archive.get_weights() # [subintegration, channel]
        nsub, npol, nchan, nbin = data.shape
        if npol < 1:
            raise ValueError("Archive contains no Stokes-I data")
        if weights.shape != (nsub, nchan):
            raise ValueError(f"Unexpected weights shape: {weights.shape}")

        intensity = data[:, 0, :, :]
        intensity = np.transpose(intensity, (1, 0, 2)).reshape(nchan, nsub * nbin)
        intensity = np.asarray(intensity, dtype=np.float32)

        # 1=retain and 0=masked. A channel must be retained in every subintegration.
        mask = np.all(weights > 0, axis=0).astype(np.uint8)

        np.save(intensity_path, intensity)
        np.save(mask_path, mask)
        converted += 1
        print(
            f"[{index:03d}/{len(archives):03d}] {archive_path.name} "
            f"I={intensity.shape} masked={int(np.sum(mask == 0))}/{nchan}",
            flush=True,
        )
    except Exception as error:
        failed.append((archive_path, error))
        print(f"[{index:03d}/{len(archives):03d}] ERROR {archive_path}: {error}")

print(f"converted_files={converted}")
print(f"failed_files={len(failed)}")
for path, error in failed:
    print(f"FAILED {path}: {error}")
