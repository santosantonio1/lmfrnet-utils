import os
import torch

from OurLMFRNet import LMFRNet

CHECKPOINT_PATH = './checkpoint/ckpt.pth'
FUSED_MODEL_PATH = './checkpoint/fused_model.pth'


def _strip_module_prefix(state_dict):
    # DataParallel checkpoints prefix every key with 'module.'
    return {
        (key[len('module.'):] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }


def _modules_to_fuse(model):
    modules_to_fuse = [[
        'features.stemBlock.stemConv.0',
        'features.stemBlock.stemConv.1',
        'features.stemBlock.stemConv.2',
    ]]

    for name, m in model.named_modules():
        if hasattr(m, 'conv') and hasattr(m, 'norm') and hasattr(m, 'relu'):
            modules_to_fuse.append([f'{name}.conv', f'{name}.norm', f'{name}.relu'])

    return modules_to_fuse


def build_fused_model(checkpoint_path=CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    weights = _strip_module_prefix(checkpoint['net'])

    model = LMFRNet()
    model.load_state_dict(weights)
    model.eval()

    fused_model = torch.quantization.fuse_modules(
        model=model,
        modules_to_fuse=_modules_to_fuse(model),
        inplace=False,
    )
    fused_model.eval()
    return fused_model


def save_fused_model(fused_model, path=FUSED_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(fused_model, path)
    print(f'[+] fused model saved to {path}')


def load_fused_model(path=FUSED_MODEL_PATH, checkpoint_path=CHECKPOINT_PATH):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    fused_model = build_fused_model(checkpoint_path)
    save_fused_model(fused_model, path)
    return fused_model


if __name__ == '__main__':
    save_fused_model(build_fused_model())
