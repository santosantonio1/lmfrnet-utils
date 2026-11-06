"""Accuracy of the float fused model vs the integer model on the CIFAR-10 test set.

    python evaluate.py --precision w16a16|w8a8
"""
import argparse

import torch

from quantize import PRECISIONS, load_images, quantize


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--precision', choices=PRECISIONS, default='w16a16')
    args = parser.parse_args()

    torch.set_num_threads(16)
    model, qm, *_ = quantize(args.precision)
    imgs, labels = load_images('test_batch')
    pf = torch.cat([model(imgs[i:i + 250]).argmax(1) for i in range(0, len(imgs), 250)])
    pq = torch.cat([qm(qm.quantize_input(imgs[i:i + 250])).argmax(1) for i in range(0, len(imgs), 250)])

    print(f'float accuracy:  {(pf == labels).float().mean() * 100:.2f}%')
    print(f'{args.precision} accuracy: {(pq == labels).float().mean() * 100:.2f}%')
    print(f'top-1 agreement: {(pf == pq).float().mean() * 100:.2f}%')


if __name__ == '__main__':
    main()
