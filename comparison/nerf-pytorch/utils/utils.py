import numpy as np
import torch


# Misc
def img2mse(x, y):
    mse = torch.mean((x - y) ** 2)
    return mse


def mse2psnr(x):
    psnr = -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
    return psnr


def to8b(x):
    img = (255 * np.clip(x, 0, 1)).astype(np.uint8)
    return img


def to8b_tensor(x):
    img = torch.tensor(255 * torch.clip(x, 0., 1.), dtype=torch.uint8)
    return img
