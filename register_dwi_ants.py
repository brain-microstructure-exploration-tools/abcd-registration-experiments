# Registration between a source and target dwi using ANTsPy (using scalar image FA) 
# We could consider multi-modal registration using different diffusivity values?
# https://github.com/ANTsX/ANTsPy

import argparse
import os
from pathlib import Path
import subprocess

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
  
print("\n  Generating dti volumes...")

script_path = Path(os.path.realpath(__file__))
generate_dti_script = '%s/reconstruct_dti.py' %(str(script_path.parent))

subprocess.run(['python', generate_dti_script, str(source_path), str(source_mask_path), str(output_path), 'source_dti.nii.gz'], stdout=subprocess.DEVNULL)
subprocess.run(['python', generate_dti_script, str(target_path), str(target_mask_path), str(output_path), 'target_dti.nii.gz'], stdout=subprocess.DEVNULL)

# === Now get into the ANTs registration ===

print("\n  Registering source fa to target fa...")

source_fa_filename = '%s/source_dti_fa.nii.gz' %(str(output_path))
target_fa_filename = '%s/target_dti_fa.nii.gz' %(str(output_path))

ants_source_im = ants.image_read(source_fa_filename)
ants_target_im = ants.image_read(target_fa_filename)

out_prefix = '%s/diffeo_' %(str(output_path))

diffeo = ants.registration(fixed=ants_target_im, moving=ants_source_im, type_of_transform='SyNRA', outprefix=out_prefix, write_composite_transform=True)

warped_image = ants.apply_transforms(fixed=ants_target_im, moving=ants_source_im, transformlist=diffeo['fwdtransforms'])

out_image_filename = '%s/warped_fa.nii.gz' %(output_path)
ants.image_write(warped_image, out_image_filename)
