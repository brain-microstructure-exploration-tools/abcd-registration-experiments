from pathlib import Path
import os
import glob

from tempfile import TemporaryDirectory

import ants
import nibabel as nib

import ants
import h5py

def register_ants_fa(source_fa, target_fa):

    ants_source_im = ants.image_read(str(source_fa))
    ants_target_im = ants.image_read(str(target_fa))

    with TemporaryDirectory() as temp_dir:

        out_prefix = '%s/diffeo_' %(str(temp_dir))

        diffeo = ants.registration(fixed=ants_target_im, moving=ants_source_im, type_of_transform='SyNRA', outprefix=out_prefix, write_composite_transform=True)
        warped_image_ants = ants.apply_transforms(fixed=ants_target_im, moving=ants_source_im, transformlist=diffeo['fwdtransforms'])
    
        # Write the warped image
        out_image_filename = '%s/warped_fa.nii.gz' %(str(temp_dir))
        ants.image_write(warped_image_ants, out_image_filename)

        warped_im_load = nib.load(out_image_filename)
        warped_im_nib = nib.Nifti1Image(warped_im_load.get_fdata(), warped_im_load.affine)

        forward_transform_path = '%s/diffeo_Composite.h5' %(str(temp_dir))
        inverse_transform_path = '%s/diffeo_InverseComposite.h5' %(str(temp_dir))

        forward_diffeo = h5py.File(forward_transform_path, 'r')
        inverse_diffeo = h5py.File(inverse_transform_path, 'r')
        
        return warped_im_nib, forward_diffeo, inverse_diffeo