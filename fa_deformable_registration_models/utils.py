import torch
import monai
import numpy as np

def mse_loss(b1, b2):
    """Return image similarity loss given two batches b1 and b2 of shape (batch_size, channels, H,W,D).
    It is scaled up a bit here."""
    return ((b1-b2)**2).mean()

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

def compose_ddf(u,v, warp):
    """Compose two displacement fields, return the displacement that warps by v followed by u
    Use the given warper (e.g. a monai.networks.blocks.Warp)"""
    return u + warp(v,u)

# Compute discrete spatial derivatives
def diff_and_trim(array, axis, spatial_size):
    """Take the discrete difference along a spatial axis, which should be 2,3, or 4.
    Return a difference tensor with all spatial axes trimmed by 1."""
    return torch.diff(array, axis=axis)[:, :, :(spatial_size[0]-1), :(spatial_size[1]-1), :(spatial_size[2]-1)]

def size_of_spatial_derivative(u):
    """Return the squared Frobenius norm of the spatial derivative of the given displacement field.
    To clarify, this is about the derivative of the actual displacement field map, not the deformation
    that the displacement field map defines. The expected input shape is (batch,3,H,W,D).
    Output shape is (batch)."""
    dx = diff_and_trim(u, 2)
    dy = diff_and_trim(u, 3)
    dz = diff_and_trim(u, 4)
    return(dx**2 + dy**2 + dz**2).sum(axis=1).mean(axis=[1,2,3])

# Monai transforms typically apply to individual objects and not batches.
# This functor takes a transform and produces a version of it that operates on batches.
# Handles MetaTensor correctly
class BatchifiedTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, batch):
        return monai.data.utils.list_data_collate(
            [self.transform(x) for x in monai.transforms.Decollated()(batch)]
        )



def _undo_crop_by_padding(crop_op, padding_mode):
    """Return a monai pad transfrom that "undoes" a crop operation by padding.

    crop_op: a description of an applied crop operation, as obtained from applied_operations
             of a monai MetaTensor
    padding_mode: see the "mode" argument of monai.transforms.Pad.

    Of course it is not possible to invert a nontrivial crop operator, but the padding operator that
    this function returns is a sort of "inverse" in the sense that applying it to the cropped image
    yields an image of the same shape as the original and with the same values as the original
    in the uncropped region. Values in the region that was cropped out are determined according to
    the chosen padding_mode.
    """
    crop_info = crop_op['extra_info']['cropped']
    to_pad = np.concatenate([np.zeros((1,2),int),np.array(crop_info).reshape((3,2))], axis=0)
    return monai.transforms.Pad(to_pad=to_pad, mode=padding_mode)

def _undo_pad_by_cropping(pad_op, spatial_dims):
    """Return a monai crop transfrom that undoes a pad operation by cropping.

    pad_op: a description of an applied pad operation, as obtained from applied_operations
            of a monai MetaTensor
    spatial_dims: the spatial dimensions of the intended input image to the return transform,
                  i.e. the spatial dimensions of the padded images whose padding we intend to undo
    """
    pad_info = pad_op['extra_info']['padded']
    if not (np.array(pad_info[0])==0).all():
        raise ValueError("_undo_pad_by_cropping does not support channel padding")
    roi_slices = [slice(p[0],s-p[1]) for p,s in zip(pad_info[1:], spatial_dims)]
    return monai.transforms.SpatialCrop(roi_slices=roi_slices)

def undo_padcrop_transform(img, padding_mode):
    """Return a monai transform that "undoes" the last pad and the last crop for the given image.

    img: a MetaTensor possibly with some applied operations in its metadata
         If it's not a MetaTensor (e.g. if it's a torch tensor) then it's assumed that there's no metadata
         to consider, so no crop or pad were ever applied and we simply return the identity transform.
    padding_mode: determines how padding should work when "undoing" a crop.
                  see the "mode" argument of monai.transforms.Pad

    Of course it is not possible to invert a nontrivial crop operator, but the padding operator that
    this function returns is a sort of "inverse" in the sense that applying it to the cropped image
    yields an image of the same shape as the original and with the same values as the original
    in the uncropped region. Values in the region that was cropped out are determined according to
    the chosen padding_mode.
    """
    if not isinstance(img, monai.data.MetaTensor):
        return monai.transforms.Identity()

    ops = img.applied_operations
    crops = list(filter(lambda o:issubclass(getattr(monai.transforms,o[1]['class']), monai.transforms.Crop),enumerate(ops)))
    pads = list(filter(lambda o:issubclass(getattr(monai.transforms,o[1]['class']), monai.transforms.Pad),enumerate(ops)))

    if len(crops)>0 and len(pads)>0:
        transforms = []
        crop_op = crops[-1][1]
        pad_op = pads[-1][1]
        assert(crops[-1][0] != pads[-1][0])
        if crops[-1][0] > pads[-1][0]: # If crop followed pad, then we uncrop and then unpad
            uncrop = _undo_crop_by_padding(crop_op, padding_mode)
            uncropped_img = uncrop(img)
            unpad = _undo_pad_by_cropping(pad_op, uncropped_img.shape[1:])
            return monai.transforms.Compose([uncrop, unpad])
        else: # If pad followed crop, then we unpad and then uncrop
            unpad = _undo_pad_by_cropping(pad_op, img.shape[1:])
            uncrop = _undo_crop_by_padding(crop_op, padding_mode)
            return monai.transforms.Compose([unpad, uncrop])


    elif len(crops)>0:
        # undo the last crop by padding
        crop_op = crops[-1][1]
        return _undo_crop_by_padding(crop_op, padding_mode)

    elif len(pads)>0:
        # undo the last pad by cropping
        pad_op = pads[-1][1]
        return _undo_pad_by_cropping(pad_op, img.shape[1:])

    else:
        return monai.transforms.Identity()