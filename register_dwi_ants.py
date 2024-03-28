# Registration between a source and target dwi using ANTsPy (using scalar image FA) 
# We could consider multi-modal registration using different diffusivity values?
# https://github.com/ANTsX/ANTsPy

import argparse
import os
from pathlib import Path
import subprocess

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

from dipy.data import get_fnames

import ants

# === Parse args ===

parser = argparse.ArgumentParser(description='performs non-linear registration between a source and target dwi')

# Positional arguments next
parser.add_argument('source', type=str, help='path to a source dwi')
parser.add_argument('source_mask', type=str, help='path to a brain mask for the source dwi')
parser.add_argument('target', type=str, help='path to a target dwi')
parser.add_argument('target_mask', type=str, help='path to a brain mask for the target dwi')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output of registration')

args = parser.parse_args()

source_path = Path(os.path.abspath(args.source))
source_without_ext = str(source_path)[:str(source_path).rfind(''.join(source_path.suffixes))]
source_bval_path = Path('%s.bval' %(source_without_ext))
source_bvec_path = Path('%s.bvec' %(source_without_ext))
source_mask_path = Path(os.path.abspath(args.source_mask))

target_path = Path(os.path.abspath(args.target))
target_without_ext = str(target_path)[:str(target_path).rfind(''.join(target_path.suffixes))]
target_bval_path = Path('%s.bval' %(target_without_ext))
target_bvec_path = Path('%s.bvec' %(target_without_ext))
target_mask_path = Path(os.path.abspath(args.target_mask))

output_path = Path(os.path.abspath(args.output_dir))

# Check source files
if not source_path.exists() or not source_bval_path.exists() or not source_bvec_path.exists():
  raise FileNotFoundError(f"File {source_path} not found (or missing corresponding bval/bvec)")
  
if not source_mask_path.exists():
  raise FileNotFoundError(f"File {source_mask_path} not found")

# Check target files  
if not target_path.exists() or not target_bval_path.exists() or not target_bvec_path.exists() :
  raise FileNotFoundError(f"File {target_path} not found (or missing corresponding bval/bvec)")
  
if not target_mask_path.exists():
  raise FileNotFoundError(f"File {target_mask_path} not found")

# If the output directory doesn't exist let's create it
if not output_path.exists():
  os.mkdir(str(output_path))
  
# Copy the masks into the output directory so we can modify/resample
working_source_mask_path = '%s/source_mask.nii.gz' %(output_path)
subprocess.run(['cp', source_mask_path, working_source_mask_path])

working_target_mask_path = '%s/target_mask.nii.gz' %(output_path)
subprocess.run(['cp', target_mask_path, working_target_mask_path])
  
# === Generate dti volumes  === (this could be made generic so we could plug in different packages/methods)

print("\n  Generating dti volumes...")

source_data, source_affine, source_img = load_nifti(str(source_path), return_img=True)
source_bvals, source_bvecs = read_bvals_bvecs(str(source_bval_path), str(source_bvec_path))

source_gtab = gradient_table(source_bvals, source_bvecs)

source_mask_data, source_mask_affine, source_mask_img = load_nifti(source_mask_path, return_img=True)

source_tenmodel = dti.TensorModel(source_gtab)
source_tenfit = source_tenmodel.fit(source_data, mask=source_mask_data)

source_dti_lotri = source_tenfit.lower_triangular()

source_out_dti_filename = '%s/source_dti.nii.gz' %(output_path)
save_nifti(source_out_dti_filename, source_dti_lotri, source_affine, source_img.header)

source_out_fa_filename = '%s/source_fa.nii.gz' %(output_path)
save_nifti(source_out_fa_filename, source_tenfit.fa, source_affine, source_img.header)

target_data, target_affine, target_img = load_nifti(str(target_path), return_img=True)
target_bvals, target_bvecs = read_bvals_bvecs(str(target_bval_path), str(target_bvec_path))

target_gtab = gradient_table(target_bvals, target_bvecs)

target_mask_data, target_mask_affine, target_mask_img = load_nifti(target_mask_path, return_img=True)

target_tenmodel = dti.TensorModel(target_gtab)
target_tenfit = target_tenmodel.fit(target_data, mask=target_mask_data)

target_dti_lotri = target_tenfit.lower_triangular()

target_out_dti_filename = '%s/target_dti.nii.gz' %(output_path)
save_nifti(target_out_dti_filename, target_dti_lotri, target_affine, target_img.header)

target_out_fa_filename = '%s/target_fa.nii.gz' %(output_path)
save_nifti(target_out_fa_filename, target_tenfit.fa, target_affine, target_img.header)

# === Now get into the ANTs registration ===

ants_source_im = ants.image_read(source_out_fa_filename)
ants_target_im = ants.image_read(target_out_fa_filename)

diffeo = ants.registration(fixed=ants_target_im, moving=ants_source_im, type_of_transform='SyNRA', outprefix='reg_', write_composite_transform=True)

print(diffeo['fwdtransforms'])

warped_image = ants.apply_transforms(fixed=ants_target_im, moving=ants_source_im, transformlist=diffeo['fwdtransforms'])

out_image_filename = '%s/warped_fa.nii.gz' %(output_path)
out_warp_filename = '%s/diffeo.nii.gz' %(output_path)
out_inverse_warp_filename = '%s/diffeo_inverse.nii.gz' %(output_path)

ants.image_write(warped_image, out_image_filename)

#warp = ants.image_read(diffeo['fwdtransforms'][0])
#ants.image_write(warp, out_warp_filename)

#warp = ants.transform_read(diffeo['fwdtransforms'][0])
#ants.image_write(warp, out_warp_filename)

#inverse_warp = ants.image_read(diffeo['invtransforms'][1])
#ants.image_write(inverse_warp, out_inverse_warp_filename)
