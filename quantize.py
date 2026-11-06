"""Quantize the fused LMFRNet to power-of-two fixed point and write C headers.

    python quantize.py --precision w16a16|w8a8        -> quantized/<precision>/

N-bit weights/activations, 2N-bit accumulators. Per conv/fc:
    acc = bias + sum(x * w)                          bias at the accumulator scale 2^(fa_in + fw)
    out = satN(acc >> s)                             s = fa_in + fw - fa_out  (vnclip, vxrm = RDN)
The bias holds the rounding term 2^(s-1), so acc >> s rounds to nearest (ties up).
then ReLU (except fc); avg_pool and gap floor, like the C kernels' shifts.

The integer model is the fused model itself, in float64 with integer weights: every
sum stays below 464*9*2^30 < 2^53, so float64 is exact.
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from fused_model import load_fused_model
from fused_weights import to_hwio
from gen_dataset import unpickle, preprocess, CLASSES

PRECISIONS = {'w16a16': (16, 32), 'w8a8': (8, 16)}  # operand bits, accumulator bits (vwmacc: SEW -> 2*SEW)
CIFAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), './autodl-tmp/data-cifar10/cifar-10-batches-py')
OUTPUT_DIR = './quantized'


def load_model():
    return load_fused_model().eval()


def load_images(batch, n=None):
    d = unpickle(f'{CIFAR_DIR}/{batch}')
    return torch.stack([preprocess(img) for img in d[b'data'][:n]]), torch.tensor(d[b'labels'][:n])


def c_name(name):
    """Module name -> the C name fused_weights.py gives its params."""
    return 'fc' if name == 'classifier' else name.replace('features.', '', 1).replace('.', '_')


def io_scales(name):
    """Names of the scales of a conv/fc's input and output activations.

    A scale is the 2^-fa that every value of a tensor is stored in; tensors that return
    the same name are fixed to the same scale. MMConv returns torch.cat([x, y1, y2, y3]),
    which in C is a plain copy (concat4_chunk) read by the next layer as a single tensor,
    so x, y1, y2 and y3 must share a scale. That output is the next layer's x, so the scale
    spans the whole stage: S<b> is stage b's input and every y1/y2/y3 in it. y0 (branch11's
    output) is not concatenated and gets a scale of its own, named after its mmLayer.
    """
    if name == 'classifier':
        return 'S5', 'logits'
    if name.startswith('features.stemBlock'):
        return 'in', 'S1'
    b = int(name.split('.')[1][-1])  # MMCBlock<b> / tran_ConvNormRelu<b>
    if 'tran' in name:
        return f'S{b}', f'S{b + 1}'
    y0 = name.split('.branch')[0]
    return {'branch11': (f'S{b}', y0), 'branch33a': (y0, f'S{b}')}.get(name.split('.')[3], (f'S{b}', f'S{b}'))


def frac_bits(max_abs, qmax):
    """Largest f such that max_abs * 2^f <= qmax."""
    return int(np.floor(np.log2(qmax / max_abs)))


@torch.no_grad()
def choose_frac_bits(model, convs, bits, acc_bits, n_calib=2000):
    """Fractional bits from the float model on n_calib train images:
    fa per activation scale (see io_scales), fw per conv/fc weights."""
    pre, post = {}, {}

    def hook(m, _, out, n=None):
        pre[n] = max(pre.get(n, 0.0), out.abs().max().item())
        post[n] = max(post.get(n, 0.0), (out.abs() if n == 'classifier' else out.clamp(min=0)).max().item())

    hooks = [m.register_forward_hook(lambda m, i, o, n=n: hook(m, i, o, n)) for n, m in convs]
    imgs, _ = load_images('data_batch_1', n_calib)
    for i in range(0, len(imgs), 500):
        model(imgs[i:i + 500])
    for h in hooks:
        h.remove()

    qmax, amax = 2**(bits - 1) - 1, 2**(acc_bits - 1) - 1
    # a scale's max is the largest max of the tensors fixed to it, so every one of them fits
    scale_max = {'in': imgs.abs().max().item()}
    for n, _ in convs:
        scale_out = io_scales(n)[1]
        scale_max[scale_out] = max(scale_max.get(scale_out, 0.0), post[n])
    fa = {k: frac_bits(v, qmax) for k, v in scale_max.items()}

    fw = {}
    for n, m in convs:
        scale_in, scale_out = io_scales(n)
        # capped at `bits` for near-zero tensors (MMCBlock4.mmLayer1.branch33c, max|w| ~ 5e-9)
        f = min(frac_bits(m.weight.abs().max().item(), qmax), bits)
        # keep 2x headroom over the calibrated accumulator, and the shift within vnclip's range
        while pre[n] * 2.0**(fa[scale_in] + f + 1) > amax or fa[scale_in] + f - fa[scale_out] > acc_bits - 1:
            f -= 1
        fw[n] = f
    return fa, fw


def build_qmodel(model, convs, fa, fw, bits, acc_bits):
    """Integer copy of the fused model: integer weights, hooks that requantize every conv/fc."""
    qm = copy.deepcopy(model).double()
    mods = dict(qm.named_modules())
    qmin, qmax = -2**(bits - 1), 2**(bits - 1) - 1
    shifts = {}
    for n, _ in convs:
        m = mods[n]
        scale_in, scale_out = io_scales(n)
        s = fa[scale_in] + fw[n] - fa[scale_out]
        assert 1 <= s < acc_bits, f'{n}: shift {s}'
        m.weight.data = torch.clamp(torch.round(m.weight.data * 2.0**fw[n]), qmin, qmax)
        # + 2^(s-1): rounds the requantize, so it is a plain shift
        m.bias.data = torch.round(m.bias.data * 2.0**(fa[scale_in] + fw[n])) + 2.0**(s - 1)
        assert m.bias.data.abs().max() < 2**(acc_bits - 1), f'{n}: bias overflow'

        def requant(_, __, acc, s=s, n=n):
            assert acc.abs().max() < 2**(acc_bits - 1), f'{n}: accumulator overflow'
            return torch.clamp(torch.floor(acc / 2.0**s), qmin, qmax)

        m.register_forward_hook(requant)
        shifts[n] = s
    for m in qm.modules():
        if isinstance(m, nn.AvgPool2d):
            m.register_forward_hook(lambda m, i, o: torch.floor(o))           # (a+b+c+d) >> 2
    mods['classifier'].register_forward_pre_hook(lambda m, i: (torch.floor(i[0]),))  # gap: sum >> 4
    qm.quantize_input = lambda x: torch.clamp(torch.round(x.double() * 2.0**fa['in']), qmin, qmax)
    return qm, shifts


def quantize(precision):
    bits, acc_bits = PRECISIONS[precision]
    model = load_model()
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    fa, fw = choose_frac_bits(model, convs, bits, acc_bits)
    qm, shifts = build_qmodel(model, convs, fa, fw, bits, acc_bits)
    return model, qm, convs, fa, fw, shifts


@torch.no_grad()
def checksums(qm, x_q):
    """Per-layer checksums of one image, named and ordered like layers[] in lmfrnet.c:
    wrapping uint32 sum of the sign-extended outputs, as PRINT_CHECKSUMS."""
    rows, mf = [], [0]

    def rec(key):
        return lambda m, i, o=None: rows.append((key, int((i[0] if o is None else o).long().sum()) % 2**32))

    def rec_mf(m, i, o):
        mf[0] += 1
        rows.append((f'mf{mf[0]}', int(o.long().sum()) % 2**32))

    mods = dict(qm.named_modules())
    hooks = [mods['features.stemBlock'].register_forward_hook(rec('stem')),
             mods['classifier'].register_forward_pre_hook(rec('tran4')),
             mods['classifier'].register_forward_hook(rec('fc'))]
    hooks += [m.register_forward_hook(rec_mf) for n, m in mods.items() if n.count('.') == 2 and '.mmLayer' in n]
    hooks += [mods[f'features.tran_pool{b}'].register_forward_hook(rec(f'tran{b}')) for b in (1, 2, 3)]
    out = qm(x_q)
    for h in hooks:
        h.remove()
    return out, rows


def write_array(path, name, ctype, values, comments):
    guard = os.path.basename(path).replace('.', '_').upper()
    with open(path, 'w') as f:
        f.write(f'#ifndef __{guard}__\n#define __{guard}__\n\n#include <stdint.h>\n\n')
        for c in comments:
            f.write(f'// {c}\n')
        f.write(f'const {ctype} {name}[{values.size}] = {{\n')
        f.write(',\n'.join(map(str, values.tolist())))
        f.write('\n};\n\n#endif\n')


@torch.no_grad()
def export(precision, root, n_images=20):
    bits, acc_bits = PRECISIONS[precision]
    _, qm, convs, fa, fw, shifts = quantize(precision)
    mods = dict(qm.named_modules())
    os.makedirs(f'{root}/params', exist_ok=True)
    os.makedirs(f'{root}/dataset', exist_ok=True)

    # weights in HWIO (fc: IO), biases at the accumulator scale
    for n, _ in convs:
        name, (scale_in, _) = c_name(n), io_scales(n)
        w = to_hwio(mods[n].weight.numpy()).astype(np.int64)
        b = mods[n].bias.numpy().astype(np.int64)
        for alias in ([name, 'classifier'] if name == 'fc' else [name]):  # lmfrnet_weights.h includes both
            write_array(f'{root}/params/{alias}_weight.h', f'{alias}_weight', f'int{bits}_t', w.flatten(),
                        [f'{alias}_weight', f'{w.shape}, Q{fw[n]}'])
            write_array(f'{root}/params/{alias}_bias.h', f'{alias}_bias', f'int{acc_bits}_t', b,
                        [f'{alias}_bias', f'{b.shape}, Q{fa[scale_in] + fw[n]} (accumulator scale), '
                                          f'+ 2^{shifts[n] - 1} to round the requantize'])

    with open(f'{root}/quant_shifts.h', 'w') as f:
        f.write('#ifndef __QUANT_SHIFTS_H__\n#define __QUANT_SHIFTS_H__\n\n')
        f.write(f'// {precision.upper()}, int{acc_bits} accumulators: '
                f'acc = bias + sum(x*w), out = sat{bits}(acc >> s)\n')
        f.write('// the bias holds the rounding term 2^(s-1): vnclip with vxrm = RDN, then ReLU (except fc)\n\n')
        f.write(f'#define INPUT_FRAC_BITS  {fa["in"]}\n')
        f.write(f'#define LOGITS_FRAC_BITS {fa["logits"]}\n\n')
        for n, _ in convs:
            scale_in, scale_out = io_scales(n)
            f.write(f'#define {c_name(n) + "_SHIFT":45s} {shifts[n]:2d} // Q{fa[scale_in]} x Q{fw[n]} -> Q{fa[scale_out]}\n')
        f.write('\n#endif\n')

    imgs, labels = load_images('test_batch', n_images)
    x_q = qm.quantize_input(imgs)
    scores, preds = qm(x_q).max(dim=1)
    nhwc = x_q.permute(0, 2, 3, 1).numpy().astype(np.int64)
    for k in range(n_images):
        label, pred, score = int(labels[k]), int(preds[k]), int(scores[k])
        write_array(f'{root}/dataset/image{k + 1}.h', f'image{k + 1}', f'int{bits}_t', nhwc[k].flatten(),
                    [f'class: {label} ({CLASSES[label]})',
                     f'predicted class: {pred} (val = {score}, {CLASSES[pred]}), {precision} model, Q{fa["logits"]}',
                     f'NHWC, Q{fa["in"]}'])

    out, rows = checksums(qm, x_q[:1])
    with open(f'{root}/golden.txt', 'w') as f:
        f.write(f'# image1, predicted class: {int(out.argmax())} (val = {int(out.max())})\nlayer,checksum\n')
        f.writelines(f'{key},{s}\n' for key, s in rows)
    print(f'[+] headers written to {root}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--precision', choices=PRECISIONS, default='w16a16')
    parser.add_argument('--output-dir', help=f'directory to write .h files to (default: {OUTPUT_DIR}/<precision>)')
    args = parser.parse_args()
    export(args.precision, args.output_dir or f'{OUTPUT_DIR}/{args.precision}')
