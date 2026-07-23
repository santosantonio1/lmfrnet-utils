import os
import torchvision

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), './autodl-tmp/data-cifar10/')

if __name__ == '__main__':
    torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
    torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
    print(f'[+] CIFAR-10 dataset ready at {DATA_ROOT}')
