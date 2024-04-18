from pathlib import Path
import subprocess
import os

from tempfile import TemporaryDirectory

import nibabel as nib
import ants

def convert_ants_transform_to_mrtrix_transform(target_image, ants_transform):

    with TemporaryDirectory() as temp_dir:

        print(temp_dir)

        identity_warp = '%s/identity_warp[].nii.gz' %(str(temp_dir))

        # Create identity warp
        subprocess.run(['warpinit', target_image, identity_warp, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transform the idenity warp using ants
        ants_target_im = ants.image_read(target_image)

        image_dim = len(ants_target_im.numpy().shape)
        
        # Transform each dimensions
        for d in range(0, image_dim):

            cur_identity_warp = '%s/identity_warp%d.nii.gz' %(temp_dir, d)

            cur_ants_source_im = ants.image_read(cur_identity_warp)
            
            cur_warped_image = ants.apply_transforms(fixed=ants_target_im, moving=cur_ants_source_im, transformlist=ants_transform, defaultvalue=2147483647)

            out_image_filename = '%s/mrtrix_warp%d.nii.gz' %(temp_dir, d)
            ants.image_write(cur_warped_image, out_image_filename)

        corrected_warp = '%s/corrected_warp.nii.gz' %(temp_dir)
        warp_filename = '%s/mrtrix_warp[].nii.gz' %(temp_dir)
        subprocess.run(['warpcorrect', warp_filename, corrected_warp, '-marker', '2147483647', '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        array_im = nib.load(corrected_warp)
        return nib.Nifti1Image(array_im.get_fdata(), array_im.affine)
