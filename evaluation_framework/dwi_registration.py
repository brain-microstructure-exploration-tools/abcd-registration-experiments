from pathlib import Path
import os
import glob

from tempfile import TemporaryDirectory

import ants
import nibabel as nib
import h5py

class RegistrationMethods:
    """
    Defines constants for registration methods
    """

    ANTS = "ants"
    VOXELMORPH = "voxelmorph"
    MRTRIX = "mrtrix"


def register_ants_fa(source_fa: Path, target_fa: Path) -> tuple[nib.nifti1.Nifti1Image, h5py.File, h5py.File]:
    """
    Performs non-linear registration between a source fa and target fa image

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