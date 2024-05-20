from pathlib import Path
import subprocess
import os

from tempfile import TemporaryDirectory

import nibabel as nib
import ants
import h5py

import voxelmorph as vxm
import tensorflow as tf

def convert_ants_transform_to_mrtrix_transform(target_image: Path, ants_transform: Path) -> nib.nifti1.Nifti1Image:
    """
    Converts an ants transformation (which is a displacement field) to an mrtrix format (which is a deformation field)
    https://community.mrtrix.org/t/registration-using-transformations-generated-from-other-packages/2259

    :param target_image: path to a target image defining the coordinate space and voxel grid
    :param ants_transform: path to an ants transform to be converted
    :return: a deformation field as a nifti image
    """

    with TemporaryDirectory() as temp_dir:

        identity_warp = '%s/identity_warp[].nii.gz' %(str(temp_dir))

        # Create identity warp
        subprocess.run(['warpinit', str(target_image), identity_warp, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transform the idenity warp using ants
        ants_target_im = ants.image_read(str(target_image))

        image_dim = len(ants_target_im.numpy().shape)
        
        # Transform each dimensions
        for d in range(0, image_dim):

            cur_identity_warp = '%s/identity_warp%d.nii.gz' %(str(temp_dir), d)

            cur_ants_source_im = ants.image_read(cur_identity_warp)
            
            cur_warped_image = ants.apply_transforms(fixed=ants_target_im, moving=cur_ants_source_im, transformlist=str(ants_transform), defaultvalue=2147483647)

            out_image_filename = '%s/mrtrix_warp%d.nii.gz' %(str(temp_dir), d)
            ants.image_write(cur_warped_image, out_image_filename)

        corrected_warp = '%s/corrected_warp.nii.gz' %(str(temp_dir))
        warp_filename = '%s/mrtrix_warp[].nii.gz' %(str(temp_dir))
        subprocess.run(['warpcorrect', warp_filename, corrected_warp, '-marker', '2147483647', '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        array_im = nib.load(corrected_warp)
        return nib.Nifti1Image(array_im.get_fdata(), array_im.affine)

def convert_voxelmorph_transform_to_mrtrix_transform(target_image: Path, voxelmorph_transform: Path, gpu: bool=False) -> nib.nifti1.Nifti1Image:
    """
    Converts an voxelmorph transformation (which is a displacement field) to an mrtrix format (which is a deformation field)
    https://community.mrtrix.org/t/registration-using-transformations-generated-from-other-packages/2259

    :param target_image: path to a target image defining the coordinate space and voxel grid
    :param voxelmorph_transform: path to a voxelmorph transform to be converted
    :return: a deformation field as a nifti image
    """
    
    # Setup device 0 = cpu, 1 = gpu
    device, nb_devices = vxm.tf.utils.setup_device(int(gpu))

    with TemporaryDirectory() as temp_dir:

        identity_warp = '%s/identity_warp[].nii.gz' %(str(temp_dir))

        # Create identity warp
        subprocess.run(['warpinit', str(target_image), identity_warp, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transform the idenity warp using voxelmorph
        voxelmorph_target_im = vxm.py.utils.load_volfile(str(target_image), add_batch_axis=True, add_feat_axis=True)
        diffeo, diffeo_affine = vxm.py.utils.load_volfile(str(voxelmorph_transform), add_batch_axis=True, ret_affine=True)

        image_dim = len(voxelmorph_target_im.shape)-2

        # Transform each dimensions
        for d in range(0, image_dim):

            cur_identity_warp = '%s/identity_warp%d.nii.gz' %(str(temp_dir), d)
            identity_im = vxm.py.utils.load_volfile(cur_identity_warp, add_batch_axis=True, add_feat_axis=True)
            
            with tf.device(device):
                cur_warped_image = vxm.networks.Transform(identity_im.shape[1:-1], interp_method='linear', nb_feats=identity_im.shape[-1]).predict([identity_im, diffeo])

            out_image_filename = '%s/mrtrix_warp%d.nii.gz' %(str(temp_dir), d)
            vxm.py.utils.save_volfile(cur_warped_image.squeeze(), out_image_filename, diffeo_affine)

        corrected_warp = '%s/corrected_warp.nii.gz' %(str(temp_dir))
        warp_filename = '%s/mrtrix_warp[].nii.gz' %(str(temp_dir))
        subprocess.run(['warpcorrect', warp_filename, corrected_warp, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        array_im = nib.load(corrected_warp)
        return nib.Nifti1Image(array_im.get_fdata(), array_im.affine)

def warp_fiber_tract(in_fiber_tract_path: Path, mrtrix_warp_path: Path, out_fiber_tract_path: Path) -> None:
    """
    Warp a tck fiber tract using mrtrix

    :param in_fiber_tract_path: path to a tck fiber tract
    :param mrtrix_warp_path: path to a deformation field
    :param out_fiber_tract_path: path to store output of the warped fiber tract 
    """

    # To be consistent with other methods, we might consider returning the tck in memory instead of writing to disk
    subprocess.run(['tcktransform', in_fiber_tract_path, mrtrix_warp_path, out_fiber_tract_path, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def warp_segmentation_image(in_segmentation_path: Path, mrtrix_warp_path: Path, out_segmentation_path: Path) -> None:
    """
    Warp a segmentation image using mrtrix

    :param in_segmentation_path: path to a nii.gz segmentation image
    :param mrtrix_warp_path: path to a deformation field
    :param out_segmentation_path: path to store output of the warped segmentation image 
    """

    subprocess.run(['mrtransform', in_segmentation_path, out_segmentation_path, '-warp', mrtrix_warp_path, '-interp', 'nearest',  '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def write_hd5_transform(hd5_file_object: h5py.File, out_name: Path) -> None:
    """
    Write an hd5 transform as a copy of a given h5py file object

    :param hd5_file_object: a h5py file object
    :param out_name: path to write the hd5 transformation
    """

    # Write the hd5 transform
    with h5py.File(str(out_name), 'w') as hd5_out_file:
        for obj in hd5_file_object.keys():
            hd5_file_object.copy(obj, hd5_out_file)