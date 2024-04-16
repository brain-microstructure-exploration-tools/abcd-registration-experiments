import monai
import torch
import torch.nn
from collections import namedtuple
from enum import Enum
from .utils import compose_ddf, size_of_spatial_derivative

ModelOutput = namedtuple("ModelOutput", "all_loss, sim_loss, icon_loss, deformation_AB")

class IconLossType(Enum):
    ICON = 0
    GRADICON = 1

# A deformable registration model
class RegNet(torch.nn.Module):
    def __init__(self,
                 lambda_reg,
                 compute_sim_loss,
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
            lambda_reg: Hyperparameter for weight of icon/gradicon loss
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
            2,  # input channels (one for fixed image and one for moving image)
            3,  # output channels (to represent 3D displacement vector field)
            channel_sequence,
            stride_sequence,
            dropout=0.2,
            norm="batch"
        )
        self.strides = stride_sequence
        self.channels = channel_sequence

        self.lambda_reg = lambda_reg
        self.compute_sim_loss = compute_sim_loss

        self.warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")

    def update_lambda_reg(self, new_lambda_reg):
        self.lambda_reg = new_lambda_reg

    def forward(self, img_A, img_B, inference_only=False):
        deformation_AB = self.reg_net(torch.cat([img_A, img_B], dim=1)) # deforms img_B to the space of img_A
        if inference_only:
            return deformation_AB
        deformation_BA = self.reg_net(torch.cat([img_B, img_A], dim=1)) # deforms img_A to the space of img_B

        img_B_warped = self.warp(img_B, deformation_AB)
        img_A_warped = self.warp(img_A, deformation_BA)
        sim_loss_A = self.compute_sim_loss(img_A, img_B_warped)
        sim_loss_B = self.compute_sim_loss(img_B, img_A_warped)
        composite_deformation_A = compose_ddf(deformation_AB, deformation_BA, self.warp)
        composite_deformation_B = compose_ddf(deformation_BA, deformation_AB, self.warp)

        if self.icon_loss_type == IconLossType.GRADICON:
            icon_loss_A = size_of_spatial_derivative(composite_deformation_A).mean()
            icon_loss_B = size_of_spatial_derivative(composite_deformation_B).mean()
        elif self.icon_loss_type == IconLossType.ICON:
            icon_loss_A = (composite_deformation_A**2).mean()
            icon_loss_B = (composite_deformation_B**2).mean()

        sim_loss = sim_loss_A + sim_loss_B
        icon_loss = icon_loss_A + icon_loss_B

        return ModelOutput(
            all_loss = sim_loss + self.lambda_reg * icon_loss,
            sim_loss = sim_loss,
            icon_loss = icon_loss,
            deformation_AB = deformation_AB
        )