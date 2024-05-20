from pathlib import Path
import os
import glob
import math

from tempfile import TemporaryDirectory

import ants
import nibabel as nib
import h5py

import numpy as np

import voxelmorph as vxm
import tensorflow as tf

from evaluation_lib import transformation_utils


class RegistrationMethods:
    """
    Defines constants for registration methods
    """

    ANTS = "ants"
    VOXELMORPH = "voxelmorph"
    MRTRIX = "mrtrix"


def zero_pad_to_multiple(im_in: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple]:
    """
    Pads a numpy array up to a given multiple (assumes 3D)

    :param im_in: the input image to pad as a numpy array
    :param multiple: the multiple to pad up to
    :return: (the image with 0 padding, a tuple of the offsets into the padded image to the original image data)
    """

    dims = im_in.shape
    pad_dims = tuple((multiple*math.ceil(d/multiple)) for d in dims)
    
    offsets = tuple(math.ceil((pd-d)/2) for pd, d in zip(pad_dims, dims))
    pad_im = np.zeros(pad_dims)
    
    pad_im[offsets[0]:offsets[0]+dims[0], offsets[1]:offsets[1]+dims[1], offsets[2]:offsets[2]+dims[2]] = im_in

    return pad_im, offsets

def register_ants_fa(source_fa: Path, target_fa: Path) -> tuple[nib.nifti1.Nifti1Image, h5py.File, h5py.File]:
    """
    Performs non-linear registration between a source fa and target fa image using ants

    :param source_fa: the source fa image in .nii.gz format
    :param target_fa: the target fa image in .nii.gz format
    :return: (warped source fa image, foward diffeo, inverse diffeo)
    """

    ants_source_im = ants.image_read(str(source_fa))
    ants_target_im = ants.image_read(str(target_fa))

    # Write ants output to a temporary directory for easy clean up
    with TemporaryDirectory() as temp_dir:

        # This is the location and prefix for saving the composite transform (affine+diffeo)
        out_prefix = '%s/diffeo_' %(str(temp_dir))

        # Run the registration
        diffeo = ants.registration(fixed=ants_target_im, moving=ants_source_im, type_of_transform='SyNRA', outprefix=out_prefix, write_composite_transform=True)
        # Apply the diffeo to the source fa image
        warped_image_ants = ants.apply_transforms(fixed=ants_target_im, moving=ants_source_im, transformlist=diffeo['fwdtransforms'])
    
        ### Read the warped image and hd5 transforms into memory as convenient object types ###

        # Write the warped image so we can read back in a more common nifti image object (rather than returning an ants image object)
        out_image_filename = '%s/warped_fa.nii.gz' %(str(temp_dir))
        ants.image_write(warped_image_ants, out_image_filename)

        warped_im_load = nib.load(out_image_filename)
        warped_im_nib = nib.Nifti1Image(warped_im_load.get_fdata(), warped_im_load.affine)

        # Read the hd5 transform files as h5py file objects for returning to the user
        forward_transform_path = '%s/diffeo_Composite.h5' %(str(temp_dir))
        inverse_transform_path = '%s/diffeo_InverseComposite.h5' %(str(temp_dir))

        forward_diffeo = h5py.File(forward_transform_path, 'r')
        inverse_diffeo = h5py.File(inverse_transform_path, 'r')

        return warped_im_nib, forward_diffeo, inverse_diffeo
    

def register_voxelmorph_fa(source_fa: Path, target_fa: Path, model_path: Path, gpu: bool=False) -> tuple[nib.nifti1.Nifti1Image, nib.nifti1.Nifti1Image]:
    """
    Performs non-linear registration between a source fa and target fa image using voxelmorph

    :param source_fa: the source fa image in .nii.gz format
    :param target_fa: the target fa image in .nii.gz format
    :param gpu: use the gpu or cpu
    :return: (warped source fa image, foward diffeo)
    """

    # Setup device 0 = cpu, 1 = gpu
    device, nb_devices = vxm.tf.utils.setup_device(int(gpu))

    # Load moving and fixed images
    moving = vxm.py.utils.load_volfile(str(source_fa), add_batch_axis=True, add_feat_axis=True)
    fixed, fixed_affine = vxm.py.utils.load_volfile(str(target_fa), add_batch_axis=True, add_feat_axis=True, ret_affine=True)

    # Voxelmorph requires the volumes be a multiple of 16 so let's zero pad the image 
    voxelmorph_multiple = 16
    
    # Get the dimensions of moving and fixed images 
    m1, l_m, w_m, h_m, m2 = moving.shape
    f1, l_f, w_f, h_f, f2 = fixed.shape

    pad_moving, offset_moving = zero_pad_to_multiple(moving[0, :, :, :, 0], voxelmorph_multiple)
    dim_pad_moving = pad_moving.shape
    new_moving = np.zeros((m1, dim_pad_moving[0], dim_pad_moving[1], dim_pad_moving[2], m2))
    new_moving[:, offset_moving[0]:offset_moving[0]+l_m, offset_moving[1]:offset_moving[1]+w_m, offset_moving[2]:offset_moving[2]+h_m, :] = moving[:, :, :, :, :]

    pad_fixed, offset_fixed = zero_pad_to_multiple(fixed[0, :, :, :, 0], voxelmorph_multiple)
    dim_pad_fixed = pad_fixed.shape
    new_fixed = np.zeros((f1, dim_pad_fixed[0], dim_pad_fixed[1], dim_pad_fixed[2], f2))
    new_fixed[:, offset_fixed[0]:offset_fixed[0]+l_f, offset_fixed[1]:offset_fixed[1]+w_f, offset_fixed[2]:offset_fixed[2]+h_f, :] = fixed[:, :, :, :, :]
    
    # Voxelmorph shaping
    inshape = new_moving.shape[1:-1]
    nb_feats = new_moving.shape[-1]

    with tf.device(device):

        # Load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(str(model_path), **config).register(new_moving, new_fixed)
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([new_moving, warp])

    # Crop out the 0 padding
    crop_warp = warp[0, offset_fixed[0]:offset_fixed[0]+l_f, offset_fixed[1]:offset_fixed[1]+w_f, offset_fixed[2]:offset_fixed[2]+h_f, :]
    crop_moved = moved[0, offset_moving[0]:offset_moving[0]+l_f, offset_moving[1]:offset_moving[1]+w_f, offset_moving[2]:offset_moving[2]+h_f, 0]

    # Write output to a temporary directory for easy clean up
    with TemporaryDirectory() as temp_dir:

        # Write the warped image and diffeo so we can read back in a more common nifti image object (rather than returning an voxelmorph image
        out_image_filename = '%s/warped_fa.nii.gz' %(str(temp_dir))
        forward_transform_path = '%s/diffeo_forward.nii.gz' %(str(temp_dir))
        
        vxm.py.utils.save_volfile(crop_moved, out_image_filename, fixed_affine)
        vxm.py.utils.save_volfile(crop_warp, forward_transform_path, fixed_affine)
        
        warped_im_load = nib.load(out_image_filename)
        warped_im_nib = nib.Nifti1Image(warped_im_load.get_fdata(), warped_im_load.affine)

        diffeo_im_load = nib.load(forward_transform_path)
        diffeo_im_nib = nib.Nifti1Image(diffeo_im_load.get_fdata(), diffeo_im_load.affine)

        return warped_im_nib, diffeo_im_nib
        

    
