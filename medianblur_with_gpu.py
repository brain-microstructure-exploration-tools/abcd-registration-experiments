# The median blur filter below is taken from https://github.com/kornia/kornia
# I modified it to work on 3D rather than 2D images, and set it to cache the kernel it uses
# I also added numpass to run the filter a number of times

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_binary_kernel3d(window_size: Tuple[int, int, int]) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches.
    If the window size is HxWxD will create a (H*W*D)xHxWxD kernel.
    """
    window_range: int = window_size[0] * window_size[1] * window_size[2]
    kernel: torch.Tensor = torch.zeros(window_range, 1, window_size[0], window_size[1], window_size[2])
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            for k in range(window_size[2]):
                l = i*window_size[2]*window_size[1]+j*window_size[2]+k
                kernel[l,0,i,j,k] += 1.0
    return kernel


def _compute_zero_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1], computed[2]


def median_blur(input: torch.Tensor, kernel_size: Tuple[int, int, int], kernel: torch.Tensor = None) -> torch.Tensor:
    r"""Blur an image using the median filter.

    Args:
        input: the input image with shape :math:`(B,C,H,W,D)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W,D)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7, 6)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7, 6])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxHxWxD. Got: {input.shape}")

    padding: Tuple[int, int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    if kernel is None:
        kernel: torch.Tensor = get_binary_kernel3d(kernel_size).to(input)
    b, c, h, w, d = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w,d)  # BxCx(K_h * K_w * K_d)xHxWxD

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median



class MedianBlur(nn.Module):
    r"""Blur an image using the median filter.

    Args:
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W, D)`
        - Output: :math:`(B, C, H, W, D)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7, 6)
        >>> blur = MedianBlur((3, 3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7, 6])
    """

    def __init__(self, kernel_size: Tuple[int, int], device='cpu') -> None:
        super().__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.kernel = get_binary_kernel3d(kernel_size).to(device)

    def forward(self, input: torch.Tensor, numpass=1) -> torch.Tensor:
        x = input
        for _ in range(numpass):
            x = median_blur(x, self.kernel_size, self.kernel)
        return x