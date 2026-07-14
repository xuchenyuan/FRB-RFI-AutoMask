# CNN--BiGRU RFI excision for FRB analysis

This repository contains a CNN--BiGRU model for predicting one RFI score for
each of 4096 frequency channels. The default threshold is `0.381270`; channels
with scores greater than or equal to this value are masked.

## Files

- `convert_data.py`: convert manually cleaned PSRCHIVE archives to NumPy data.
- `model.py`: CNN--BiGRU model definition.
- `data.py`: NumPy dataset and normalization.
- `train.py`: train the model.
- `test.py`: evaluate a checkpoint and optionally export it to ONNX.
- `apply_mask_onnx.cc`: apply the ONNX model to a PSRCHIVE archive on a CPU.

## Environment

The Python programs require Python 3.10 or newer, NumPy and PyTorch. ONNX is
also required when exporting the trained model:

```bash
python -m pip install numpy torch onnx
```

`convert_data.py` and the C++ inference program require a working PSRCHIVE
installation. PSRCHIVE has several external dependencies and is not installed
by the command above; follow the PSRCHIVE installation instructions for the
local system.

The C++ program additionally requires the ONNX Runtime C/C++ library. Download
the Linux x64 release archive from:

<https://github.com/microsoft/onnxruntime/releases>

Extract it to a local directory before compiling the program.

## Preparing labelled data

The input files must be folded PSRCHIVE archives that have already been
manually inspected with `pazi`. The channel weights stored by `pazi` are used
as the reference masks: a positive weight means retain the channel and a zero
weight means mask it.

Place the archives directly in one input directory:

```text
archives/
├── burst1.ar.dm
├── burst2.ar.dm
└── burst3.dm.pazi
```

Edit `INPUT_DIR`, `OUTPUT_DIR` and `ARCHIVE_SUFFIXES` at the top of
`convert_data.py`, then run:

```bash
python convert_data.py
```

For every archive the script writes:

```text
data/converted/example.I.npy       # float32, [4096, 4096]
data/converted/example.mask.npy    # uint8, [4096], 1=retain and 0=mask
```

The converter does not choose a training/test split. After conversion, manually
move each complete `.I.npy`/`.mask.npy` pair into `data/train` or `data/test`.
Keep related observations in the same split and choose the split before model
training or evaluation.

The path settings in the Python files may be absolute or relative. Relative
paths are interpreted from the directory in which the script is run. The code
also calls `expanduser()`, so paths beginning with `~/` are supported.

## Training

Edit the settings at the top of `train.py`, especially `TRAIN_DIR`,
`CHECKPOINT_DIR` and `BATCH_SIZE`, then run:

```bash
python train.py
```

The default schedule uses AdamW for 35 epochs, an initial learning rate of
`1e-3`, weight decay `1e-2`, and learning-rate reductions by a factor of 0.3
after epochs 20 and 30.

`train.py` reads samples from disk as needed instead of loading the complete
dataset into memory. It uses seed `1`, an independent random-order generator
and fixed CuDNN settings. The reported train loss is recomputed over the
training split after every epoch. Exact floating-point values can still depend
on the PyTorch, CUDA and cuDNN versions and the hardware.

The script saves every epoch as `checkpoints/cnn_bigru_epoch_XX.pth` and
records the training loss in `training_log.csv`.

## Evaluation and ONNX export

Edit `TEST_DIR`, `MODEL_FILE`, `THRESHOLD` and the other settings at the top of
`test.py`, then run:

```bash
python test.py
```

The program prints ROC AUC, mean TP/FN/TN/FP channel counts per file,
bad-channel recall, good-channel retention and mean forward time. It does not
make plots or write modified archives.

To export the model, set:

```python
EXPORT_ONNX = True
ONNX_FILE = Path("cnn_bigru.onnx")
```

The exported model accepts a fixed `[1, 4096, 4096]` float32 tensor and returns
bad-channel probabilities with shape `[1, 4096]`.

## Compiling the C++ inference program

Download and extract the ONNX Runtime C/C++ Linux x64 release from
<https://github.com/microsoft/onnxruntime/releases>. The program also requires
a working PSRCHIVE installation. Set `ORT` to the extracted ONNX Runtime
directory and `PSR` to the PSRCHIVE installation prefix before compiling:

```bash
export ORT=/path/to/onnxruntime
export PSR=/path/to/psrchive

PSR_INC="-I$PSR/include -I$PSR/include/psrchive"
PSR_INC="$PSR_INC $(find $PSR/include/psrchive -type d | awk '{printf "-I%s ", $0}')"

g++ -O3 -march=native -mtune=native -std=c++17 \
  apply_mask_onnx.cc -o apply_mask_onnx \
  $PSR_INC \
  -I$ORT/include \
  -L$PSR/lib \
  -L$ORT/lib \
  -lonnxruntime \
  -lpsrmore \
  -lpsrbase \
  -lpsrutil \
  -lcfitsio \
  -lfftw3f \
  -lfftw3 \
  -pthread \
  -ldl \
  -lm \
  -Wl,--allow-shlib-undefined \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  -Wl,-rpath,$PSR/lib \
  -Wl,-rpath,$ORT/lib
```

For a PSRCHIVE installation inside the active Conda environment, use:

```bash
export PSR=$CONDA_PREFIX
```

The two rpath options allow the executable to locate the PSRCHIVE and ONNX
Runtime shared libraries without changing `LD_LIBRARY_PATH`.

## Applying a mask

Use the default threshold and one CPU thread:

```bash
./apply_mask_onnx input.ar cnn_bigru.onnx output.ar
```

The threshold and ONNX Runtime intra-op thread count can be specified:

```bash
./apply_mask_onnx input.ar cnn_bigru.onnx output.ar 0.38127 4
```

The input archive is not modified. Pre-existing zero channel weights are
preserved, and the new mask is written to the output archive.
