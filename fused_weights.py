import argparse
import os
import numpy as np

from fused_model import load_fused_model

QUANTIZATION_SCALE = 8192
OUTPUT_DIR = './fused-weights'


def quantize(params):
    return params.detach().numpy() * QUANTIZATION_SCALE


def to_hwio(params):
    if params.ndim == 4:
        return params.transpose(2, 3, 1, 0)
    if params.ndim == 2:
        return params.T
    return params


def generate_header(name, params, root=OUTPUT_DIR, layout='hwio'):
    os.makedirs(root, exist_ok=True)

    if layout == 'hwio':
        params = to_hwio(params)
        
    name = name.replace('features.', '', 1).replace('.', '_')
    header_guard = name.upper()
    values = params.flatten().astype(np.int32)

    filename = f'{root}/{name}.h'
    with open(filename, 'w') as f:
        f.write(f'#ifndef __{header_guard}_H__\n')
        f.write(f'#define __{header_guard}_H__\n\n')
        f.write(f'// {name}\n')
        f.write(f'// {params.shape}\n')
        f.write(f'const int {name}[] = {{\n')
        f.write(',\n'.join(map(str, values)))
        f.write('\n};\n')
        f.write('\n#endif')

    print(f'[+] parameters generated in {filename} file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantized fused-model weights as .h files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='directory to write .h files to')
    parser.add_argument('--layout', choices=['hwio', 'oihw'], default='hwio',
                        help='weight order: hwio (TensorFlow) or oihw (PyTorch)')
    args = parser.parse_args()

    fused_model = load_fused_model()

    for name, params in fused_model.state_dict().items():
        generate_header(name, quantize(params), root=args.output_dir, layout=args.layout)
