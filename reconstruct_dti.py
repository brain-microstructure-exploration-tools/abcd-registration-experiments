# Reconstruct the diffusion tensor image given a dwi
# Assumes there is a correspondingly named .bval and .bvec in same folder as source

import argparse
import os
from pathlib import Path
import subprocess

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

from dipy.data import get_fnames

# === Parse args ===

parser = argparse.ArgumentParser(description='Reconstructs dti from given dti')

# Optional arguments
parser.add_argument('--save_fa', default=1, type=int, help='save the fa image')

# Positional arguments next
parser.add_argument('dwi', type=str, help='path to a source dwi')
parser.add_argument('dwi_mask', type=str, help='path to a brain mask for the source dwi')
parser.add_argument('output_dir', type=str, help='path to a folder for saving dti')
parser.add_argument('output_name', type=str, help='filename for the output dti')

args = parser.parse_args()

save_fa = args.save_fa

dwi_path = Path(os.path.abspath(args.dwi))
dwi_without_ext = str(dwi_path)[:str(dwi_path).rfind(''.join(dwi_path.suffixes))]
dwi_bval_path = Path('%s.bval' %(dwi_without_ext))
dwi_bvec_path = Path('%s.bvec' %(dwi_without_ext))
dwi_mask_path = Path(os.path.abspath(args.dwi_mask))

output_path = Path(os.path.abspath(args.output_dir))
output_name = args.output_name

# Check source files
if not dwi_path.exists() or not dwi_bval_path.exists() or not dwi_bvec_path.exists():
  raise FileNotFoundError(f"File {dwi_path} not found (or missing corresponding bval/bvec)")
  
if not dwi_mask_path.exists():
  raise FileNotFoundError(f"File {dwi_mask_path} not found")

# If the output directory doesn't exist let's create it
if not output_path.exists():
  os.mkdir(str(output_path))

# === Generate dti volumes  === (this could be made generic so we could plug in different packages/methods)

dwi_data, dwi_affine, dwi_img = load_nifti(str(dwi_path), return_img=True)
dwi_bvals, dwi_bvecs = read_bvals_bvecs(str(dwi_bval_path), str(dwi_bvec_path))

dwi_gtab = gradient_table(dwi_bvals, dwi_bvecs)

dwi_mask_data, dwi_mask_affine, dwi_mask_img = load_nifti(dwi_mask_path, return_img=True)

tenmodel = dti.TensorModel(dwi_gtab)
tenfit = tenmodel.fit(dwi_data, mask=dwi_mask_data)

dti_lotri = tenfit.lower_triangular()

out_dti_filename = Path('%s/%s' %(output_path, output_name))
save_nifti(str(out_dti_filename), dti_lotri, dwi_affine, dwi_img.header)

if (save_fa):
    filename_without_ext = str(out_dti_filename)[:str(out_dti_filename).rfind(''.join(out_dti_filename.suffixes))]
    out_fa_filename = '%s_fa.nii.gz' %(filename_without_ext)
    save_nifti(out_fa_filename, tenfit.fa, dwi_affine, dwi_img.header)
