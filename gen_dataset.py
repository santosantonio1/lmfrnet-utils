import argparse
import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

from fused_model import load_fused_model

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    './autodl-tmp/data-cifar10/cifar-10-batches-py/test_batch',
)
OUTPUT_DIR = './dataset-quant'
QUANTIZATION_SCALE = 8192

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])


def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def preprocess(image):
    img = image.reshape(3, 32, 32).transpose(1, 2, 0)
    return transform(img)


def quantize(img):
    return (img.unsqueeze(0).permute(0, 2, 3, 1).numpy().flatten() * QUANTIZATION_SCALE).astype(np.int32)


def generate_header(name, values, label, prediction, root=OUTPUT_DIR):
    os.makedirs(root, exist_ok=True)

    filename = f'{root}/{name}.h'
    header_guard = name.upper()
    with open(filename, 'w') as f:
        f.write(f'#ifndef __{header_guard}_H__\n')
        f.write(f'#define __{header_guard}_H__\n\n')
        f.write(f'// class: {label} ({CLASSES[label]})\n')
        f.write(f'// predicted class: {prediction} (val = {prediction}, {CLASSES[prediction]})\n')
        f.write(f'const int {name}[{values.size}] = {{\n')
        f.write(',\n'.join(map(str, values)))
        f.write('\n};\n')
        f.write('\n#endif')

    print(f'[+] image generated in {filename} file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantized CIFAR-10 test images as .h files')
    parser.add_argument('--num-images', type=int, default=10, help='number of images to generate')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='directory to write .h files to')
    args = parser.parse_args()

    test_batch = unpickle(DATA_PATH)
    labels = test_batch[b'labels'][:args.num_images]
    imgs = torch.stack([preprocess(image) for image in test_batch[b'data'][:args.num_images]])

    model = load_fused_model()
    with torch.no_grad():
        predictions = model(imgs).argmax(dim=1).tolist()

    for n, (img, label, prediction) in enumerate(zip(imgs, labels, predictions), start=1):
        generate_header(f'image{n}', quantize(img), label, prediction, root=args.output_dir)
