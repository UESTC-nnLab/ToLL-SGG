import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_pool2d(x: torch.Tensor, kernel_size, stride=None):
    """
    Minimal SoftPool2D implementation.
    For the common case in this repo: kernel_size == [H, W] (global soft pooling),
    it performs a softmax-weighted average over spatial dims.
    Otherwise, it falls back to avg_pool2d to keep things runnable.
    """
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])

    b, c, h, w = x.shape

    # Global soft pooling (what generator.py uses: kernel_size=[G,1])
    if kh == h and kw == w and (stride is None or stride == (1, 1)):
        x_flat = x.reshape(b, c, -1)
        weights = torch.softmax(x_flat, dim=-1)
        y = (weights * x_flat).sum(dim=-1, keepdim=True)  # (B,C,1)
        return y.view(b, c, 1, 1)

    # Fallback (rarely hit in this repo)
    return F.avg_pool2d(x, kernel_size=(kh, kw), stride=stride)


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
