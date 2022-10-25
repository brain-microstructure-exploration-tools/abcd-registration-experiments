import monai
import torch
import torch.nn
from enum import Enum
from collections import namedtuple
from pathlib import Path


state_dict_path = Path(__file__).parent / '2022-09-13-deformable-2be0f3bd.pth'
if not state_dict_path.exists():
  raise FileNotFoundError(f"Make sure to download the model weights so that {state_dict_path} is found.")

ModelOutput = namedtuple("ModelOutput", "all_loss, sim_loss, icon_loss, deformation_AB")

warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")

def compose_ddf(u,v):
    """Compose two displacement fields, return the displacement that warps by v followed by u"""
    return u + warp(v,u)

class IconLossType(Enum):
    ICON = 0
    GRADICON = 1

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

def ncc_loss(b1, b2):
    """Return the negative NCC loss given two batches b1 and b2 of shape (batch_size, channels, H,W,D).
    It is averaged over batches and channels."""
    mu1 = b1.mean(dim=(2,3,4)) # means
    mu2 = b2.mean(dim=(2,3,4))
    alpha1 = (b1**2).mean(dim=(2,3,4)) # second moments
    alpha2 = (b2**2).mean(dim=(2,3,4))
    alpha12 = (b1*b2).mean(dim=(2,3,4)) # cross term
    numerator = alpha12 - mu1*mu2
    denominator = torch.sqrt((alpha1 - mu1**2) * (alpha2-mu2**2))
    ncc = numerator / denominator
    return -ncc.mean() # average over batches and channels

# A deformable registration model
class RegModel(torch.nn.Module):
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

    def update_lambda_reg(self, new_lambda_reg):
        self.lambda_reg = new_lambda_reg

    def multiscale_reg_nets(self, img_A, img_B, cache_multiscale_warps=False):
        """
        Here we expect img_A to be a list consisting of batches of target images:
            img_A[0] is a batch of target images at the original resolution,
            img_A[1] is a batch of target images downsampled by a factor of 2 in each dimension,
            img_A[2] is a batch of target images downsampled by a factor of 4 in each dimension,
            etc.
        and similarly img_B is a list consisting of batches of moving images.
        Returns the final displacement field (composed over all scales) for deforming img_B[0] to img_A[0].

        cache_multiscale_warps: Whether to cache the last-computed warps at each scale
        in attributes phis and phi_comps.
        phis[i] is the warp at scale i, where i=0 is the original image scale.
        """

        if cache_multiscale_warps:
            self.phis=[None]*self.num_subnetworks
            self.phi_comps=[None]*self.num_subnetworks

        for i in range(self.num_subnetworks - 1, -1, -1): # Run backwards to 0 from num_subnetworks-1

            # phi_up = Composite of warps up to scale i+1, operating at scale i
            if i==self.num_subnetworks - 1:
                pass # Base case: phi_up is the identity map.
                # (We treat this case specially below to avoid complicating the computational graph with
                # useless compositions with identity map)
            else:
                phi_up = self.batch_resizers[i](phi_comp)

            # warped_B = img_B at scale i warped by the composite of warps up to scale i+1
            if i==self.num_subnetworks - 1:
                warped_B = img_B[i] # Base case: "the composite of warps up to scale i+1" = the identity map
            else:
                warped_B = warp(img_B[i], phi_up)

            # phi = Warp from scale i, operating at scale i
            phi = self.reg_nets[i](torch.cat([img_A[i], warped_B], dim=1))

            # phi_comp = Composite of warps up to scale i, operating at scale i
            if i==self.num_subnetworks - 1:
                phi_comp = phi # Base case: phi_up = the identity map, i.e. chain to compose consists of phi only
            else:
                phi_comp = compose_ddf(phi,phi_up)

            if cache_multiscale_warps:
                self.phis[i] = phi
                self.phi_comps[i] = phi_comp

        return phi_comp

    def forward_inference(self, img_A, img_B):
        deformation_AB = self.reg_net(torch.cat([img_A, img_B], dim=1)) # deforms img_B to the space of img_A
        img_B_warped = warp(img_B, deformation_AB)
        return deformation_AB, img_B_warped

    def forward_train(self, img_A, img_B):
        """For reference, we keep this function which was the old forward function used during training"""
        deformation_AB = self.reg_net(torch.cat([img_A, img_B], dim=1)) # deforms img_B to the space of img_A
        deformation_BA = self.reg_net(torch.cat([img_B, img_A], dim=1)) # deforms img_A to the space of img_B

        img_B_warped = warp(img_B, deformation_AB)
        img_A_warped = warp(img_A, deformation_BA)
        sim_loss_A = self.compute_sim_loss(img_A, img_B_warped)
        sim_loss_B = self.compute_sim_loss(img_B, img_A_warped)
        composite_deformation_A = compose_ddf(deformation_AB, deformation_BA)
        composite_deformation_B = compose_ddf(deformation_BA, deformation_AB)

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

def create_model(device):
    model = RegModel(
        lambda_reg = 0.05,
        compute_sim_loss = ncc_loss,
        down_convolutions=4,
        depth=4,
        max_channels=256,
        init_channels=32,
        icon_loss_type=IconLossType.ICON,
    ).to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.load_state_dict(torch.load(state_dict_path))

    return model

  # TODO then test this and make sure it works, then strip out losses and stuff, keeping only warp