# Registration between a source and target dti using dti-tk (requires dti-tk installed and setup per instructions)
# https://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Install

import argparse
import os
import subprocess
from pathlib import Path

# === Parse args ===

parser = argparse.ArgumentParser(description='performs non-linear registration between a source and target dwi')

# Optional arguments
parser.add_argument('--scale', default=1875, type=int, help='scaling to correct for diffusivity units')
parser.add_argument('--iters', default=6, type=int, help='number of iterations for diffeomorphic registration')
parser.add_argument('--tol', default=0.002, type=float, help='convergence critera for diffeomorphic registration')

# Positional arguments next
parser.add_argument('source', type=str, help='path to a source dti')
parser.add_argument('source_mask', type=str, help='path to a brain mask for the source dwi')
parser.add_argument('target', type=str, help='path to a target dti')
parser.add_argument('target_mask', type=str, help='path to a brain mask for the target dwi')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output of registration')

args = parser.parse_args()

diffusivity_scale = args.scale
num_iters = args.iters
tolerance = args.tol

source_path = Path(os.path.abspath(args.source))
source_mask_path = Path(os.path.abspath(args.source_mask))

target_path = Path(os.path.abspath(args.target))
target_mask_path = Path(os.path.abspath(args.target_mask))

output_path = Path(os.path.abspath(args.output_dir))

# Check source files
if not source_path.exists():
  raise FileNotFoundError(f"File {source_path} not found")
  
if not source_mask_path.exists():
  raise FileNotFoundError(f"File {source_mask_path} not found")

# Check target files  
if not target_path.exists():
  raise FileNotFoundError(f"File {target_path} not found")
  
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

# === Preprocess dti for registration with dti-tk ===

print("\n  Preprocessing dti volumes...")

source_out_filename = '%s/%s' %(str(output_path), str(source_path.stem))
target_out_filename = '%s/%s' %(str(output_path), str(target_path.stem))

# First we have to scale the diffusivity values to make the units compatible with dti-tk (mean diffusivity for CSF around 3)
# For now I have checked a couple of subjects and found MD to be around 0.0016 --> so I set the scale to 1875 as default
# This should be revisited for the population and for different scanners (possible not hardcoded if we have tissue seg)
subprocess.run(['TVtool', '-in', source_path, '-scale', str(diffusivity_scale), '-out', source_out_filename], stdout=subprocess.DEVNULL)
subprocess.run(['TVtool', '-in', target_path, '-scale', str(diffusivity_scale), '-out', target_out_filename], stdout=subprocess.DEVNULL)

# Set origin to [0,0,0] as recommended by dti-tk documentation
subprocess.run(['TVAdjustVoxelspace', '-in', source_out_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
subprocess.run(['SVAdjustVoxelspace', '-in', working_source_mask_path, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
subprocess.run(['TVAdjustVoxelspace', '-in', target_out_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
subprocess.run(['SVAdjustVoxelspace', '-in', working_target_mask_path, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)

# Dimensions must be powers of 2 as per dti-tk documentation (resample voxel size accordingly)
subprocess.run(['TVResample', '-in', source_out_filename, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)
subprocess.run(['SVResample', '-in', working_source_mask_path, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)
subprocess.run(['TVResample', '-in', target_out_filename, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)
subprocess.run(['SVResample', '-in', working_target_mask_path, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)

# === Now exceute rigid, affine, and non-linear registration with dti-tk ===

print("\n  Beginning registration...")
print("    Rigid...")

rigid_log_path = '%s/rigid_log' %(output_path)
rigid_log = open(rigid_log_path, 'w')
subprocess.run(['dti_rigid_reg', target_out_filename, source_out_filename, 'EDS', '4', '4', '4', '0.01'], stdout=rigid_log) 

print("    Affine...")

affine_log_path = '%s/affine_log' %(output_path)
affine_log = open(affine_log_path, 'w')
subprocess.run(['dti_affine_reg', target_out_filename, source_out_filename, 'EDS', '4', '4', '4', '0.01', '1'], stdout=affine_log) 

print("    Diffeomorphic...")

diffeo_log_path = '%s/diffeo_log' %(output_path)
diffeo_log = open(diffeo_log_path, 'w')
rootname = source_out_filename[:source_out_filename.rfind(''.join(Path(source_out_filename).suffixes))]
affine_filename = '%s_aff.nii.gz' %(rootname)

print(affine_filename)

subprocess.run(['dti_diffeomorphic_reg', target_out_filename, affine_filename, working_target_mask_path, '1', str(num_iters), str(tolerance)], stdout=diffeo_log)

print()
