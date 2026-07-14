"""Evaluate CNN--BiGRU and optionally export it to ONNX."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BurstMaskDataset
from model import (
    CNNBiGRU,
    DEFAULT_THRESHOLD,
    N_CHANNELS,
    N_TIME_BINS,
    ScoreModel,
    VALID_FIRST,
    VALID_LAST,
)


# =========================
# Settings
# =========================

TEST_DIR = Path("data/test").expanduser()
MODEL_FILE = Path("cnn_bigru.pth").expanduser()
THRESHOLD = DEFAULT_THRESHOLD

BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPORT_ONNX = False
ONNX_FILE = Path("cnn_bigru.onnx").expanduser()


def auc_score(labels, scores):
    """ROC AUC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


device = torch.device(DEVICE)
dataset = BurstMaskDataset(TEST_DIR)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)

model = CNNBiGRU().to(device)
model.load_state_dict(
    torch.load(MODEL_FILE, map_location=device, weights_only=True)
)
model.eval()

all_scores = []
all_labels = []
forward_seconds = 0.0

with torch.no_grad():
    for intensity, target, _ in loader:
        intensity = intensity.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        probability = torch.sigmoid(model(intensity))
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds += time.time() - start

        all_scores.append(
            probability[:, VALID_FIRST:VALID_LAST].cpu().numpy().reshape(-1)
        )
        all_labels.append(
            target[:, VALID_FIRST:VALID_LAST].numpy().reshape(-1)
        )

scores = np.concatenate(all_scores)
labels = np.concatenate(all_labels).astype(bool)
prediction = scores >= THRESHOLD

tp = int(np.sum(prediction & labels))
fn = int(np.sum(~prediction & labels))
tn = int(np.sum(~prediction & ~labels))
fp = int(np.sum(prediction & ~labels))
n_files = len(dataset)

print(f"files={n_files}")
print(f"threshold={THRESHOLD:.6f}")
print(f"auc={auc_score(labels, scores):.6f}")
print(f"mean_tp_per_file={tp / n_files:.2f}")
print(f"mean_fn_per_file={fn / n_files:.2f}")
print(f"mean_tn_per_file={tn / n_files:.2f}")
print(f"mean_fp_per_file={fp / n_files:.2f}")
print(f"bad_recall={tp / (tp + fn):.6f}")
print(f"good_retention={tn / (tn + fp):.6f}")
print(f"mean_forward_seconds={forward_seconds / n_files:.6f}")

if EXPORT_ONNX:
    ONNX_FILE.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ScoreModel(model).to(device).eval()
    example = torch.zeros(1, N_CHANNELS, N_TIME_BINS, device=device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            ONNX_FILE,
            opset_version=17,
            input_names=["input"],
            output_names=["score"],
            do_constant_folding=True,
        )
    print(f"onnx_model={ONNX_FILE}")
