import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np


def percent_negative_jacobian_determinant(diffeo_path: Path) -> float:
    '''
    Computes the percentage of voxels that have a negative Jacobian determinant, which indicates local non-diffeomorphic behavior.
    Uses mrtrix command warp2metric

    :param diffeo_path: path to an mrtrix warp in a nii.gz format 
    :return: the percentage of voxels with negative Jacobian determinant
    '''

    with TemporaryDirectory() as temp_dir: 

        jacobian_determinant_image_path = Path(temp_dir) / 'jac_det.nii.gz'

        subprocess.run(['warp2metric', str(diffeo_path), '-jdet', jacobian_determinant_image_path, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        jacobian_determinant_image_nib = nib.load(str(jacobian_determinant_image_path))
        jacobian_determinant_image = np.array(jacobian_determinant_image_nib.dataobj)

        return float((jacobian_determinant_image < 0).sum() / jacobian_determinant_image.size)
