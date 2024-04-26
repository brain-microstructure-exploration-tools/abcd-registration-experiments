# Evaluates ants pairwise registration between a source and a target dwi via fa images
# Currently there are several temporary files created and not deleted, and hardcoded filenames -- this may be good enough for our own evaluation purposes -- can revisit
# The source_fiber_dir and target_fiber_dir assume the same number of fiber tracts with the same names in tck format

import argparse
from pathlib import Path
import subprocess
import os
import glob

import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
import h5py

import process_dwi
import pairwise_evaluation
import dwi_registration
import transformation_utils

parser = argparse.ArgumentParser(description='Performs pairwise dwi registration using ANTS and evaluates the accuracy via fiber tract distance (more metrics to be implemented)')

# Optional arguments
parser.add_argument('--percent_sample_fibers', default=0.1, type=float, help='randomly sample a percentage of fiber streamlines')
parser.add_argument('--num_repeats', type=int, help='the number of times to repeat the fiber tract distance measurement')

# Positional arguments next
parser.add_argument('source', type=str, help='path to a source dwi')
parser.add_argument('source_mask', type=str, help='path to a brain mask for the source dwi')
parser.add_argument('source_fiber_dir', type=str, help='directory for source fiber tracts')
parser.add_argument('target', type=str, help='path to a target dwi')
parser.add_argument('target_mask', type=str, help='path to a brain mask for the target dwi')
parser.add_argument('target_fiber_dir', type=str, help='directory for target fiber tracts')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output of registration')

args = parser.parse_args()

percent_sample_fibers = args.percent_sample_fibers

source_path = Path(os.path.abspath(args.source))
source_without_ext = str(source_path)[:str(source_path).rfind(''.join(source_path.suffixes))]
source_bval_path = Path('%s.bval' %(source_without_ext))
source_bvec_path = Path('%s.bvec' %(source_without_ext))
source_mask_path = Path(os.path.abspath(args.source_mask))
source_fiber_path = Path(os.path.abspath(args.source_fiber_dir))

target_path = Path(os.path.abspath(args.target))
target_without_ext = str(target_path)[:str(target_path).rfind(''.join(target_path.suffixes))]
target_bval_path = Path('%s.bval' %(target_without_ext))
target_bvec_path = Path('%s.bvec' %(target_without_ext))
target_mask_path = Path(os.path.abspath(args.target_mask))
target_fiber_path = Path(os.path.abspath(args.target_fiber_dir))

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

# === Generate dti and fa images ===

print("\n  Processing dwi and computing fa images...")

source_dwi_info, source_dti_lotri, source_fa_image = process_dwi.reconstruct_dti(source_path, source_bval_path, source_bvec_path, source_mask_path)
target_dwi_info, target_dti_lotri, target_fa_image = process_dwi.reconstruct_dti(target_path, target_bval_path, target_bvec_path, target_mask_path)

# Save the fa images
out_source_fa_filename = Path('%s/source_fa.nii.gz' %(output_path))
out_target_fa_filename = Path('%s/target_fa.nii.gz' %(output_path))
save_nifti(str(out_source_fa_filename), source_fa_image, source_dwi_info['affine'], source_dwi_info['header'])
save_nifti(str(out_target_fa_filename), target_fa_image, target_dwi_info['affine'], target_dwi_info['header'])

print("\n  Running ants registration...")

warped_fa, forward_diffeo, inverse_diffeo = dwi_registration.register_ants_fa(out_source_fa_filename, out_target_fa_filename)

# Write deformed fa
out_image_path = Path('%s/warped_fa.nii.gz' %(str(output_path)))
nib.save(warped_fa, str(out_image_path))

# Write hd5 transforms
forward_diffeo_filename = Path('%s/diffeo_Composite.h5' %(str(output_path)))
inverse_diffeo_filename = Path('%s/diffeo_InverseComposite.h5' %(str(output_path)))

transformation_utils.write_hd5_transform(forward_diffeo, forward_diffeo_filename)
transformation_utils.write_hd5_transform(inverse_diffeo, inverse_diffeo_filename)

print("\n  Running ants evalutation...")

pairwise_evaluation.pairwise_evaluation_ants(out_target_fa_filename, forward_diffeo_filename, inverse_diffeo_filename, source_fiber_path, target_fiber_path, output_path)

