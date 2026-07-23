# LMFRNet

This repo builds on [wgqhandsome/LMFRNet](https://github.com/wgqhandsome/LMFRNet/tree/main),
a lightweight, DenseNet-inspired CNN trained on CIFAR-10. It exists to support bare-metal
C experiments with LMFRNet: tooling here fuses and quantizes the trained model, and
quantizes CIFAR-10 test images, into fixed-point C header files.

## Project layout

| File | Purpose |
|---|---|
| `OurLMFRNet.py` | Model definition (`LMFRNet`, `MMBlock`/`MMConv`, `StemBlock`), adapted from the original repo so its Conv+BatchNorm+ReLU layers are structured for `torch.quantization.fuse_modules` to fuse. |
| `download_dataset.py` | Downloads CIFAR-10 into `autodl-tmp/data-cifar10/`. |
| `fused_model.py` | Loads `checkpoint/ckpt.pth`, fuses Conv+BatchNorm+ReLU layers, and caches the result to `checkpoint/fused_model.pth`. |
| `fused_weights.py` | Exports the fused model's quantized weights as `.h` files. |
| `gen_dataset.py` | Exports quantized CIFAR-10 test images as `.h` files. |
| `checkpoint/ckpt.pth` | Pretrained weights (tracked in git). |

Generated artifacts (`checkpoint/fused_model.pth`, `fused-weights/`, `dataset-quant/`,
`autodl-tmp/`) are gitignored — they're all reproducible from the scripts above.

## Setup

```sh
pip install torch torchvision numpy
```

## Usage

**1. Get the dataset** (only needed for `gen_dataset.py`):

```sh
python download_dataset.py
```

**2. Export quantized model weights as C headers:**

```sh
python fused_weights.py [--output-dir ./fused-weights]
```

Fuses Conv+BatchNorm+ReLU in `checkpoint/ckpt.pth` (caching to
`checkpoint/fused_model.pth` on first run) and writes one `.h` file per
parameter tensor, quantized to fixed-point `int32` (×8192).

**3. Export quantized test images as C headers:**

```sh
python gen_dataset.py [--num-images 10] [--output-dir ./dataset-quant]
```

Writes `image1.h` .. `imageN.h`, each holding a normalized, fixed-point
`int32` (×8192) CIFAR-10 test image, in the same quantization scheme as the
weights above — for feeding into the embedded inference code alongside the
exported weights.

## Quantization scheme

Both weights and images are scaled by `8192` (`QUANTIZATION_SCALE` in
`fused_weights.py`/`gen_dataset.py`) and cast to `int32`, matching a
fixed-point inference implementation on the embedded target.
