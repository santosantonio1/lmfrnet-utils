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
| `quantize.py` | Quantizes the fused model to W16A16 or W8A8 fixed point with 2N-bit accumulators and exports weights, shifts and images as `.h` files, plus golden checksums. |
| `evaluate.py` | Compares the accuracy of the fused float model and the `quantize.py` integer model on the CIFAR-10 test set. |
| `checkpoint/ckpt.pth` | Pretrained weights (tracked in git). |

Generated artifacts (`checkpoint/fused_model.pth`, `fused-weights/`, `dataset-quant/`,
`quantized/`, `autodl-tmp/`) are gitignored — they're all reproducible from the scripts above.

## Setup

```sh
pip install torch torchvision numpy
```

## Usage

**1. Get the dataset** (needed for `gen_dataset.py`, `quantize.py` and `evaluate.py`):

```sh
python download_dataset.py
```

**2. Export quantized model weights as C headers:**

```sh
python fused_weights.py [--output-dir ./fused-weights] [--layout hwio|oihw]
```

Fuses Conv+BatchNorm+ReLU in `checkpoint/ckpt.pth` (caching to
`checkpoint/fused_model.pth` on first run) and writes one `.h` file per
parameter tensor, quantized to fixed-point `int32` (×8192).
`--layout` picks the weight order: `hwio` (default) or `oihw`, PyTorch's native order.

**3. Export quantized test images as C headers:**

```sh
python gen_dataset.py [--num-images 10] [--output-dir ./dataset-quant]
```

Writes `image1.h` .. `imageN.h`, each holding a normalized, fixed-point
`int32` (×8192) CIFAR-10 test image, in the same quantization scheme as the
weights above — for feeding into the embedded inference code alongside the
exported weights. Each image is also permuted from PyTorch's NCHW layout to
TensorFlow's NHWC layout before flattening, and the header comments the true
class and the fused model's predicted class (index + name) for that image.

**4. Export a W16A16 or W8A8 model as C headers:**

```sh
python quantize.py [--precision w16a16|w8a8] [--output-dir ./quantized/<precision>]
python evaluate.py [--precision w16a16|w8a8]
```

`quantize.py` writes `params/*.h` (`intN_t` weights in HWIO, `int2N_t` biases),
`quant_shifts.h` (one requantization shift per layer), `dataset/image1.h` ..
`image20.h` (`intN_t`, NHWC) and `golden.txt` (per-layer checksums of image1, in
the order `PRINT_CHECKSUMS` in the C app prints them). `evaluate.py` prints the float and
integer accuracy and their top-1 agreement on the 10k test images.

## Quantization scheme

`fused_weights.py` and `gen_dataset.py` scale both weights and images by `8192`
(`QUANTIZATION_SCALE`) and cast to `int32`, matching a fixed-point inference
implementation on the embedded target.

`quantize.py` uses power-of-two scales chosen per tensor: a value `v` is stored as
`round(v * 2^f)` in N bits (16 or 8), with `f` fractional bits per layer's weights
and per activation scale, calibrated on 2000 training images. Each conv/fc
accumulates in 2N bits, starting at the bias, then shifts back to N bits with saturation
(`vnclip` with `vxrm = RDN`), so the RISC-V vector kernels can use widening MACs. The
bias holds the rounding term `2^(s-1)`, so the shift rounds to nearest.

| | Test accuracy |
|---|---|
| float (fused model) | 94.54% |
| W16A16, int32 accumulators | 94.55% |
| W8A8, int16 accumulators | 93.71% |
