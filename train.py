"""Train the CNN--BiGRU frequency-channel RFI mask model."""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import BurstMaskDataset
from model import CNNBiGRU, VALID_FIRST, VALID_LAST


# =========================
# Settings
# =========================

TRAIN_DIR = Path("data/train").expanduser()
CHECKPOINT_DIR = Path("checkpoints").expanduser()
LOG_FILE = Path("training_log.csv").expanduser()

EPOCHS = 35
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Fix the Python, NumPy, PyTorch and sample-order random generators.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

device = torch.device(DEVICE)
dataset = BurstMaskDataset(TRAIN_DIR)
valid = slice(VALID_FIRST, VALID_LAST)

# Determine the class weight without retaining the arrays in memory.
n_bad = 0.0
n_good = 0.0
for index in range(len(dataset)):
    _, target, _ = dataset[index]
    target = target[valid]
    n_bad += float((target > 0.5).sum())
    n_good += float((target <= 0.5).sum())
pos_weight = n_good / max(n_bad, 1.0)

model = CNNBiGRU().to(device)
bce = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 30], gamma=0.3
)

# Use the same independent CPU generator for the sample order.
rng = torch.Generator(device="cpu")
rng.manual_seed(SEED)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def weighted_loss(logits, target):
    loss = bce(logits, target)
    weight = torch.ones_like(target)
    weight[target > 0.5] = pos_weight
    return (loss[:, valid] * weight[:, valid]).mean()


def load_batch(indices):
    """Load one batch from disk in the requested order."""
    intensities = []
    targets = []
    for index in indices.tolist():
        intensity, target, _ = dataset[int(index)]
        intensities.append(intensity)
        targets.append(target)
    return torch.stack(intensities), torch.stack(targets)


def evaluate_train_loss():
    """Recompute loss over the training split after an epoch."""
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            indices = torch.arange(start, min(start + BATCH_SIZE, len(dataset)))
            xb, yb = load_batch(indices)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            losses.append(float(weighted_loss(model(xb), yb).item()))
    return float(np.mean(losses))


print(
    f"device={device} files={len(dataset)} "
    f"parameters={sum(p.numel() for p in model.parameters())} "
    f"pos_weight={pos_weight:.6f} seed={SEED}"
)

training_start = time.time()
with LOG_FILE.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["epoch", "learning_rate", "train_loss", "epoch_seconds"])

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        order = torch.randperm(len(dataset), generator=rng)

        for start in range(0, len(dataset), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            xb, yb = load_batch(indices)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_train_loss()
        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        checkpoint_file = CHECKPOINT_DIR / f"cnn_bigru_epoch_{epoch:02d}.pth"
        torch.save(model.state_dict(), checkpoint_file)

        writer.writerow([epoch, current_lr, train_loss, epoch_seconds])
        handle.flush()
        print(
            f"epoch={epoch:02d}/{EPOCHS} lr={current_lr:.3e} "
            f"train_loss={train_loss:.6f} time={epoch_seconds:.1f}s "
            f"checkpoint={checkpoint_file}",
            flush=True,
        )
        scheduler.step()

print(f"saved_checkpoints={CHECKPOINT_DIR}")
print(f"saved_log={LOG_FILE}")
print(f"total_seconds={time.time() - training_start:.1f}")
