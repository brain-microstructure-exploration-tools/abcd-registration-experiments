from .reg_net1 import RegNet, IconLossType
from .utils import ncc_loss, BatchifiedTransform, undo_padcrop_transform
import torch
import monai
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
trained_state_dict_path = os.path.join(package_directory, 'reg_model1.pth')

class RegModel:
  def __init__(self,device=None):
    if device is None:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.spatial_size = (144,144,144)
    self.device = device

    self.model = RegNet(
        lambda_reg = 0.05, # for the record, this was ramped up on a schedule during training
        compute_sim_loss = ncc_loss,
        down_convolutions=4,
        depth=4,
        max_channels=256,
        init_channels=32,
        icon_loss_type=IconLossType.ICON,
    ).to(device)

    self.model.load_state_dict(torch.load(trained_state_dict_path, map_location=self.device))
    self.model.eval()

    self.transform = monai.transforms.Compose([
      monai.transforms.CenterSpatialCrop(roi_size=self.spatial_size),
      monai.transforms.SpatialPad(spatial_size=self.spatial_size, mode="constant"),
      monai.transforms.ToDevice(device=self.device),
    ])

    self.transform_batch = BatchifiedTransform(self.transform)
    self.collate = monai.data.utils.list_data_collate
    self.decollate = monai.transforms.Decollated()

  def forward(self, target, moving, include_warped_image=False, use_input_coords=True):
    """Pass in FA images as tensors of shape (1,H,W,D). They will be cropped down to and/or padded up
    to spatial dimensions (144,144,144) in order to be passed into the network.

    Returns displacement field, and if include_warped_image is enabled then also the warped moving image.

    The returned displacement field and warped image are in the same coordinates as the input images,
    unless use_input_coords is turned off, in which case the returned displacement field and warped image
    are in the native coordinates of the underlying predictive network, i.e. coordinates in which
    spatial size is (144,144,144).
    """
    target_batch = self.collate([target])
    moving_batch = self.collate([moving])
    if include_warped_image:
      deformation_AB_batch, warped_batch = self.forward_batch(
        target_batch, moving_batch,
        include_warped_image=include_warped_image,
        use_input_coords=use_input_coords
      )
      return self.decollate(deformation_AB_batch)[0], self.decollate(warped_batch)[0]
    else:
      deformation_AB_batch = self.forward_batch(
        target_batch, moving_batch,
        include_warped_image=include_warped_image,
        use_input_coords=use_input_coords
      )
      return self.decollate(deformation_AB_batch)[0]

  def forward_batch(self, target, moving, include_warped_image=False, use_input_coords=True):
    """Pass in a batch of FA images as tensors of shape (b,1,H,W,D). They will be cropped down to and/or padded up
    to spatial dimensions (144,144,144).

    Returns displacement fields, and if include_warped_image is enabled then also the warped moving images.

    The returned displacement fields and warped images are in the same coordinates as the input images,
    unless use_input_coords is turned off, in which case the returned displacement fields and warped images
    are in the native coordinates of the underlying predictive network, i.e. coordinates in which
    spatial size is (144,144,144)."""
    target_in = self.transform_batch(target)
    moving_in = self.transform_batch(moving)
    with torch.no_grad():
      deformation_AB = self.model(target_in, moving_in, inference_only=True)

    if use_input_coords:
      undo_padcrop = undo_padcrop_transform(self.decollate(moving_in)[0], padding_mode="edge")
      undo_padcrop_batchified = BatchifiedTransform(undo_padcrop)
      deformation_AB = undo_padcrop_batchified(deformation_AB)
      thing_to_warp = moving
    else:
      thing_to_warp = moving_in

    if include_warped_image:
      warped = self.model.warp(thing_to_warp, deformation_AB)
      return deformation_AB, warped
    else:
      return deformation_AB