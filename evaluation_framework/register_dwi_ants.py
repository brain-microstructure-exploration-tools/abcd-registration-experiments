# Registration between a source and target dwi using ANTsPy (using scalar image FA) 
# We could consider multi-modal registration using different diffusivity values?
# https://github.com/ANTsX/ANTsPy

import argparse
import os
from pathlib import Path
import subprocess

import ants
from dipy.io.image import save_nifti

import process_dwi

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
  
print("\n  Generating dti volumes...")

source_dwi_info, source_dti_lotri, source_fa_image = process_dwi.reconstruct_dti(source_path, source_bval_path, source_bvec_path, source_mask_path)
target_dwi_info, target_dti_lotri, target_fa_image = process_dwi.reconstruct_dti(target_path, target_bval_path, target_bvec_path, target_mask_path)

# Save the fa images
out_source_fa_filename = Path('%s/source_fa.nii.gz' %(output_path))
out_target_fa_filename = Path('%s/target_fa.nii.gz' %(output_path))
save_nifti(str(out_source_fa_filename), source_fa_image, source_dwi_info['affine'], source_dwi_info['header'])
save_nifti(str(out_target_fa_filename), target_fa_image, target_dwi_info['affine'], target_dwi_info['header'])

# === Now get into the ANTs registration ===

print("\n  Registering source fa to target fa...")

ants_source_im = ants.image_read(str(out_source_fa_filename))
ants_target_im = ants.image_read(str(out_target_fa_filename))

out_prefix = '%s/diffeo_' %(str(output_path))

diffeo = ants.registration(fixed=ants_target_im, moving=ants_source_im, type_of_transform='SyNRA', outprefix=out_prefix, write_composite_transform=True)

warped_image = ants.apply_transforms(fixed=ants_target_im, moving=ants_source_im, transformlist=diffeo['fwdtransforms'])

out_image_filename = '%s/warped_fa.nii.gz' %(output_path)
ants.image_write(warped_image, out_image_filename)
