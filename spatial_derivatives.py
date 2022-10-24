from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_kernel_bias() -> torch.Tensor:
    """Create sobel style kernel and bias for computing spatial derivatives (via central difference).
    This is specifically for use with the spatial_derivative function defined below."""
    kernel: torch.Tensor = torch.zeros(3,1,3,3,3)

    # sobel x
    kernel[0,0,0,1,1] -= 0.5
    kernel[0,0,2,1,1] += 0.5

    # sobel y
    kernel[1,0,1,0,1] -= 0.5
    kernel[1,0,1,2,1] += 0.5

    # sobel z
    kernel[2,0,1,1,0] -= 0.5
    kernel[2,0,1,1,2] += 0.5

    # The kernel constructed above can be used to get the spatial gradient of a scalar function, shape (b,1,H,W,D)

    # The bias is what we need to add to each channel in order to go from looking at the derivative of the
    # displacement field to looking at the derivative of the transformation
    bias = torch.eye(3,3, dtype=kernel.dtype).reshape(-1)

    return kernel, bias

def spatial_derivative(input: torch.Tensor, kernel_bias: Tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the spatial derivative of a 3D transformation represented as a displacement vector field.

    Args:
        input: the input image representing a spatial transformation as a displacement vector field;
            should have shape :math:`(B,3,H,W,D)`
        kernel: optionally a prebuilt kernel and bias to use
             kernel shape: :math:`(9,1,3,3,3)`
             bias shape:   :math:`(9,)`

    Returns:
        the jacobian matrix field with shape :math:`(B,9,H,W,D)`,
        where the entry (b,c,x,y,z) has the following interpretation for image at batch index b
        located at (x,y,z) at each value of i:
            i=0: x-derivative of of the x-component of the transformation
            i=1: y-derivative of of the x-component of the transformation
            i=2: z-derivative of of the x-component of the transformation
            i=3: x-derivative of of the y-component of the transformation
            i=4: y-derivative of of the y-component of the transformation
            i=5: z-derivative of of the y-component of the transformation
            i=6: x-derivative of of the z-component of the transformation
            i=7: y-derivative of of the z-component of the transformation
            i=8: z-derivative of of the z-component of the transformation

    If you reshape the return value to (B,3,3,H,W,D) then you will have 3x3 jacobian matrices
    living in the dimensions 1 and 2. That is, [b,:,:,x,y,z] will be the 3x3 jacobian matrix
    for the transformation with batch index b at location (x,y,z).
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5 or input.shape[1]!=3:
        raise ValueError(f"Invalid input shape, we expect Bx3xHxWxD. Got: {input.shape}")


    if kernel_bias is None:
        kernel, bias = get_kernel_bias().to(input)
    else:
        kernel, bias = kernel_bias

    b,c,h,w,d = input.shape
    assert(c==3)
    deriv: torch.Tensor = F.conv3d(input.view(b*c,1,h,w,d), kernel, stride=1, groups=1)
    deriv = F.pad(deriv, (1,1,1,1,1,1), mode='replicate')
    deriv = deriv.view(b,3*c,h,w,d)
    deriv += bias.view(1,9,1,1,1)


    return deriv



class DerivativeOfDDF(nn.Module):
    r"""Compute the spatial derivative of a 3D transformation represented as a displacement vector field.

    We use central difference. In order to keep spatial dimensions the same,
    the boundary is padded in "replicate" mode, so the derivatives on the boundary
    are off by one voxel. That is, the derivatives are first computed to produce a smaller image,
    and then replication padding fixes the image size.
    This is not ideal (a better approach would be to use forward-difference
    or backward difference at the boundaries), but this is much simpler, and it shouldn't
    matter very much (it matters only when the displacement field has large second derivatives
    at the boundary *and* when what happens at the boundary is actually important-- this situation
    is just not very likely for my use case.)

    Args:
        input: the input image representing a spatial transformation

    Returns:
        the jacobian matrix field with shape :math:`(B,9,H,W,D)`,
        where the entry (b,c,x,y,z) has the following interpretation for image at batch index b
        located at (x,y,z) at each value of i:
            i=0: x-derivative of of the x-component of the transformation
            i=1: y-derivative of of the x-component of the transformation
            i=2: z-derivative of of the x-component of the transformation
            i=3: x-derivative of of the y-component of the transformation
            i=4: y-derivative of of the y-component of the transformation
            i=5: z-derivative of of the y-component of the transformation
            i=6: x-derivative of of the z-component of the transformation
            i=7: y-derivative of of the z-component of the transformation
            i=8: z-derivative of of the z-component of the transformation

    Shape:
        - Input: :math:`(B, 3, H, W, D)`
        - Output: :math:`(B, 9, H, W, D)`

    If you reshape the ouput to (B,3,3,H,W,D) then you will have 3x3 jacobian matrices
    living in the dimensions 1 and 2. That is, [b,:,:,x,y,z] will be the 3x3 jacobian matrix
    for the transformation with batch index b at location (x,y,z).
    """

    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.kernel, self.bias = get_kernel_bias()
        self.kernel = self.kernel.to(device)
        self.bias = self.bias.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_derivative(input, (self.kernel, self.bias))