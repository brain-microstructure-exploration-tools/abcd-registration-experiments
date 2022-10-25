import monai
import torch
import torch.nn
from enum import Enum
from collections import namedtuple
from pathlib import Path
from typing import Union
from dti_warp import WarpDTI, MseLossDTI, TensorTransformType, PolarDecompositionMode
from util import ComposeDDF


state_dict_path = Path(__file__).parent / 'dti-2022-10-20b-e17e67307.pth'
if not state_dict_path.exists():
  raise FileNotFoundError(f"Make sure to download the model weights so that {state_dict_path} is found.")


spatial_size = (144,144,144)

H, W, D = spatial_size

# Compute discrete spatial derivatives
def diff_and_trim(array, axis):
    """Take the discrete difference along a spatial axis, which should be 2,3, or 4.
    Return a difference tensor with all spatial axes trimmed by 1."""
    return torch.diff(array, axis=axis)[:, :, :(H-1), :(W-1), :(D-1)]

def size_of_spatial_derivative(u):
    """Return the squared Frobenius norm of the spatial derivative of the given displacement field.
    To clarify, this is about the derivative of the actual displacement field map, not the deformation
    that the displacement field map defines. The expected input shape is (batch,3,H,W,D).
    Output shape is (batch)."""
    dx = diff_and_trim(u, 2)
    dy = diff_and_trim(u, 3)
    dz = diff_and_trim(u, 4)
    return(dx**2 + dy**2 + dz**2).sum(axis=1).mean(axis=[1,2,3])

ModelOutput = namedtuple("ModelOutput", "all_loss, sim_loss, icon_loss, deformation_AB, sim_loss_weighted, icon_loss_weighted")

class IconLossType(Enum):
    ICON = 0
    GRADICON = 1

# A deformable registration model
class RegModel(torch.nn.Module):
    def __init__(self,
                 device,
                 lambda_reg,
                 lambda_sim,
                 down_convolutions,
                 depth,
                 max_channels,
                 init_channels,
                 icon_loss_type : IconLossType = IconLossType.GRADICON,
                 downsample_early = True
                ):
        """
        Create a deformable registration network

        Args:
            device:
            lambda_reg: Hyperparameter for weight of icon/gradicon loss
            lambda_sim: Hyperparameter for weight of similarity loss (not independent from lambda_reg so
                not really a new hyperparameter)
            compute_sim_loss: A function that compares two batches of images and returns a similarity loss
            down_convolutions: How many stride=2 convolutions to include in the down-convolution part of the unets
                               when at the original image scale. We assume the original image size is divisible by
                               2**down_convolutions
            depth: Total number of layers in half of the unet. Increase this to increase model capacity.
                   Must be >= down_convolutions
            max_channels: As you go to deeper layers, channels grow by powers of two... up to a maximum given here.
            init_channels: how many channels in the first layer
            icon_loss_type: whether to use ICON or GradICON
            downsample_early: The CNN can contain strided and unstrided convolutions to achieve the requested
                              depth; this paramter decides whether to prefer putting strided convolutions earlier
                              or later.
        """
        super().__init__()
        self.icon_loss_type = icon_loss_type
        if depth < down_convolutions:
            raise ValueError("Must have depth >= down_convolutions")
        # (We will assume that the original image size is divisible by 2**n.)



        num_twos = down_convolutions # The number of 2's we will put in the sequence of convolutional strides.
        num_ones = depth-down_convolutions # The number of 1's
        num_one_two_pairs = min(num_ones, num_twos) # The number of 1,2 pairs to stick in the middle
        if downsample_early:
            stride_sequence = (2,)*(num_twos-num_one_two_pairs) + (1,2)*num_one_two_pairs + (1,)*(num_ones-num_one_two_pairs)
        else:
            stride_sequence = (1,)*(num_ones-num_one_two_pairs) + (1,2)*num_one_two_pairs + (2,)*(num_twos-num_one_two_pairs)
        channel_sequence = [min(init_channels*2**c,max_channels) for c in range(num_twos+num_ones+1)]

        self.reg_net = monai.networks.nets.UNet(
            3,  # spatial dims
            12, # input channels (6 for lower triangular entries of fixed image and 6 for moving image)
            3,  # output channels (to represent 3D displacement vector field)
            channel_sequence,
            stride_sequence,
            dropout=0., # achieving ICON with dropout seems not to always be preserved when dropout is removed...
            norm="instance"
        )

        # Zero-initialize last upconv weights so that we start with zero ddf (thus perfect ICON score)
        last_layer = self.reg_net.model[-1][0]
        with torch.no_grad():
            last_layer.weight.zero_()
            last_layer.bias.zero_()


        self.strides = stride_sequence
        self.channels = channel_sequence

        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.compute_sim_loss = MseLossDTI(device=device)

        self.warp = WarpDTI(
            device = device,
            tensor_transform_type = TensorTransformType.FINITE_STRAIN,
            polar_decomposition_mode = PolarDecompositionMode.HALLEY,
            num_iterations = 3,
        )
        self.compose_ddf = ComposeDDF()
        self.to(device)

    def update_lambda_reg(self, new_lambda_reg):
        self.lambda_reg = new_lambda_reg

    def forward(self, img_A, img_B, return_warp_only = False,  img_A_fa = None, img_B_fa = None) -> Union[ModelOutput,torch.Tensor]:
        deformation_AB = self.reg_net(torch.cat([img_A, img_B], dim=1)) # deforms img_B to the space of img_A
        if return_warp_only:
            return deformation_AB
        deformation_BA = self.reg_net(torch.cat([img_B, img_A], dim=1)) # deforms img_A to the space of img_B

        img_B_warped = self.warp(img_B, deformation_AB)
        img_A_warped = self.warp(img_A, deformation_BA)
        if img_A_fa is not None and img_B_fa is not None: # If FA images are provided, we weight MSE by them
            sim_loss_A = self.compute_sim_loss(img_A, img_B_warped, img_A_fa*0.5+0.5)
            sim_loss_B = self.compute_sim_loss(img_B, img_A_warped, img_B_fa*0.5+0.5)
        else:
            sim_loss_A = self.compute_sim_loss(img_A, img_B_warped)
            sim_loss_B = self.compute_sim_loss(img_B, img_A_warped)
        composite_deformation_A = self.compose_ddf(deformation_AB, deformation_BA)
        composite_deformation_B = self.compose_ddf(deformation_BA, deformation_AB)

        if self.icon_loss_type == IconLossType.GRADICON:
            icon_loss_A = size_of_spatial_derivative(composite_deformation_A).mean()
            icon_loss_B = size_of_spatial_derivative(composite_deformation_B).mean()
        elif self.icon_loss_type == IconLossType.ICON:
            icon_loss_A = (composite_deformation_A**2).mean()
            icon_loss_B = (composite_deformation_B**2).mean()

        sim_loss = sim_loss_A + sim_loss_B
        icon_loss = icon_loss_A + icon_loss_B

        return ModelOutput(
            all_loss = self.lambda_sim * sim_loss + self.lambda_reg * icon_loss,
            sim_loss = sim_loss,
            icon_loss = icon_loss,
            sim_loss_weighted = self.lambda_sim * sim_loss,
            icon_loss_weighted = self.lambda_reg * icon_loss,
            deformation_AB = deformation_AB
        )



def create_model(device):
    model = RegModel(
        lambda_reg = 2,
        lambda_sim = 1e7,
        device=device,
        down_convolutions=4,
        depth=4,
        max_channels=256,
        init_channels=32,
        icon_loss_type=IconLossType.ICON,
    ).to(device)

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.load_state_dict(torch.load(state_dict_path))

    return model