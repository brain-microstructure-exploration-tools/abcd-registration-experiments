import math
import operator
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import ants
import h5py
import nibabel as nib
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


class RegistrationMethods:
    """
    Defines constants for registration methods
    """

    ANTS = "ants"
    VOXELMORPH = "voxelmorph"
    MRTRIX = "mrtrix"
    DTITK = 'dtitk'

def crop_to_size(im_in: np.ndarray, crop_size: Tuple) -> np.ndarray:
    '''
    Crops a numpy array to a given size (centered crop)
    :param im_in: the input image to crop as a numpy array
    :param crop_size: tuple determining the crop size
    :return: a numpy array of the desired size
    '''

    start = tuple(map(lambda a, da: a//2-da//2, im_in.shape, crop_size))
    end = tuple(map(operator.add, start, crop_size))
    slices = tuple(map(slice, start, end))

    return im_in[slices]

def zero_pad_to_multiple(im_in: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple]:
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

def register_ants_fa(source_fa: Path, target_fa: Path) -> Tuple[nib.nifti1.Nifti1Image, h5py.File, h5py.File]:
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


def register_voxelmorph_fa(source_fa: Path, target_fa: Path, model_path: Path, gpu: bool=False) -> Tuple[nib.nifti1.Nifti1Image, nib.nifti1.Nifti1Image]:
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


def register_dtitk_dti(source_path: Path, target_path: Path, target_mask_path: Path, diffusivity_scale: int=1500) -> nib.nifti1.Nifti1Image:
    """
    Performs non-linear registration between a source dti and target dti image using dtitk

    :param source_path: the source dti image in .nii.gz format
    :param target_path: the target dti image in .nii.gz format
    :param target_mask_path: the target brain mask image in .nii.gz format
    :param diffusivity_scale: scale the diffusivity values to make the units compatible with dtitk
    :return: (foward diffeo)
    """

    # Hardcoded centered crop size that has been confirmed to work for the ABCD evaluation set
    crop_size = (128, 128, 128)

    # Hardcoded recommended values for registration (could be also be passed in but I doubt we'll ever explore these)
    num_iters = 6
    tolerance = 0.002

    # Write output to a temporary directory for easy clean up
    with TemporaryDirectory() as temp_dir:

        # Filenames for the 128x128x128 cropped images
        source_out_dti_filename = '%s/source_dti.nii.gz' %(temp_dir)
        target_out_dti_filename = '%s/target_dti.nii.gz' %(temp_dir)
        target_out_mask_filename = '%s/target_mask.nii.gz' %(temp_dir)

        ### Source crop and save ###

        source_dti_nifti = nib.load(source_path)
        source_dti = source_dti_nifti.get_fdata()
        source_dti_pad = crop_to_size(source_dti, crop_size)

        # Update header
        source_dti_nifti.header['dim'] = [4, crop_size[0], crop_size[1], crop_size[2], 6, 1, 1, 1]
        source_dti_pad_nib = nib.Nifti1Image(source_dti_pad, source_dti_nifti.affine, header=source_dti_nifti.header)
        nib.save(source_dti_pad_nib, source_out_dti_filename)

        ### Target crop and save ###

        target_dti_nifti = nib.load(target_path)
        target_dti = target_dti_nifti.get_fdata()
        target_dti_pad = crop_to_size(target_dti, crop_size)

        # Update header
        target_dti_nifti.header['dim'] = [4, crop_size[0], crop_size[1], crop_size[2], 6, 1, 1, 1]
        target_dti_pad_nib = nib.Nifti1Image(target_dti_pad, target_dti_nifti.affine, header=target_dti_nifti.header)
        nib.save(target_dti_pad_nib, target_out_dti_filename)

        # ### Target mask crop and save ###

        target_mask_nifti = nib.load(target_mask_path)
        target_mask = target_mask_nifti.get_fdata()
        target_mask_pad = crop_to_size(target_mask, crop_size)

        target_mask_size = target_mask.shape

        # Update header
        target_mask_nifti.header['dim'] = [3, crop_size[0], crop_size[1], crop_size[2], 1, 1, 1, 1]
        target_mask_pad_nib = nib.Nifti1Image(target_mask_pad, target_mask_nifti.affine, header=target_mask_nifti.header)
        nib.save(target_mask_pad_nib, target_out_mask_filename)

        # First we have to scale the diffusivity values to make the units compatible with dti-tk (mean diffusivity for CSF around 3)
        # For now I have checked a couple of subjects and found MD to be around 0.002 --> so I set the scale to 1500 as default
        # This should be revisited for the population and for different scanners (possible not hardcoded if we have tissue seg)
        subprocess.run(['TVtool', '-in', source_out_dti_filename, '-scale', str(diffusivity_scale), '-out', source_out_dti_filename], stdout=subprocess.DEVNULL)
        subprocess.run(['TVtool', '-in', target_out_dti_filename, '-scale', str(diffusivity_scale), '-out', target_out_dti_filename], stdout=subprocess.DEVNULL)

        # Set origin to [0,0,0] as recommended by dti-tk documentation
        subprocess.run(['TVAdjustVoxelspace', '-in', source_out_dti_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
        subprocess.run(['TVAdjustVoxelspace', '-in', target_out_dti_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
        subprocess.run(['SVAdjustVoxelspace', '-in', target_out_mask_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)

        ### Run the registration ###

        # Rigid
        subprocess.run(['dti_rigid_reg', target_out_dti_filename, source_out_dti_filename, 'EDS', '4', '4', '4', '0.01'], stdout=subprocess.DEVNULL)

        # Affine
        subprocess.run(['dti_affine_reg', target_out_dti_filename, source_out_dti_filename, 'EDS', '4', '4', '4', '0.01', '1'], stdout=subprocess.DEVNULL)

        # Diffeomorphic
        affine_filename = '%s/source_dti_aff.nii.gz' %(temp_dir)
        subprocess.run(['dti_diffeomorphic_reg', target_out_dti_filename, affine_filename, target_out_mask_filename, '1', str(num_iters), str(tolerance)])#, stdout=subprocess.DEVNULL)

        affine_trans = '%s/source_dti.aff' %(temp_dir)
        diffeo_trans = '%s/source_dti_aff_diffeo.df.nii.gz' %(temp_dir)
        combined_trans = '%s/combined.nii' %(temp_dir)
        # Now combine the affine and diffeo into the overall transformation
        subprocess.run(['dfRightComposeAffine', '-aff', affine_trans, '-df', diffeo_trans, '-out', combined_trans], stdout=subprocess.DEVNULL)

        ### Registration is over -- hack the diffeo back to the original coordinate space ###

        # Crazy hacky part -- load the diffeo and add the ridiculous extra dimension to make things work, because.
        diffeo_nib = nib.load(combined_trans)
        diffeo_np = diffeo_nib.get_fdata()
        the_shape = diffeo_np.shape
        diffeo_np = np.reshape(diffeo_np, (the_shape[0], the_shape[1], the_shape[2], 1, 3))
        out_diffeo_nib = nib.Nifti1Image(diffeo_np, diffeo_nib.affine, header=diffeo_nib.header)
        nib.save(out_diffeo_nib, combined_trans)

        # Resample to the crop size
        crop_string = '%dx%dx%d' %(crop_size[0], crop_size[1], crop_size[2])
        resampled_trans = '%s/combined_resampled.nii' %(temp_dir)
        # Resample to the diffeo to the size of the input (128x128x128 with 1.7x1.7x1.7 spacing)
        subprocess.run(['c3d', '-mcs', combined_trans, '-foreach', '-resample', crop_string, '-endfor', '-omc', '3', resampled_trans], stdout=subprocess.DEVNULL)

        # Load the diffeo and uncrop it back to 140x140x140
        diffeo_nib = nib.load(resampled_trans)
        diffeo_np = diffeo_nib.get_fdata()

        out_diffeo_np = np.zeros((target_mask_size[0], target_mask_size[1], target_mask_size[2], 1, 3), dtype=diffeo_np.dtype)
        offsetx = (target_mask_size[0]-crop_size[0])//2
        offsety = (target_mask_size[1]-crop_size[1])//2
        offsetz = (target_mask_size[2]-crop_size[2])//2

        resampled_resized_trans = '%s/combined_resampled_resized.nii' %(temp_dir)
        out_diffeo_np[offsetx:offsetx+crop_size[0], offsety:offsety+crop_size[1], offsetz:offsetz++crop_size[2], :, :] = diffeo_np

        # Modify header and save it
        diffeo_nib.header['intent_code'] = '1007'
        out_diffeo_nib = nib.Nifti1Image(out_diffeo_np, diffeo_nib.affine, header=diffeo_nib.header)
        nib.save(out_diffeo_nib, resampled_resized_trans)

        # Reorient and fix the origin
        final_diffeo = '%s/final_diffeo.nii' %(temp_dir)
        subprocess.run(['c3d', '-mcs', resampled_resized_trans, '-foreach', '-orient', 'RPI', '-origin', '-119x117.3x-117.3mm', '-endfor', '-omc', '3', final_diffeo], stdout=subprocess.DEVNULL)

        # Read in the temp file and return it to the user
        diffeo_nib = nib.load(final_diffeo)
        diffeo_np = diffeo_nib.get_fdata().squeeze()
        out_diffeo_nib = nib.Nifti1Image(diffeo_np, diffeo_nib.affine, header=diffeo_nib.header)

        return out_diffeo_nib