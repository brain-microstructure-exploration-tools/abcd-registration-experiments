# Evaluates ants pairwise registration between a source and a target dwi via fa images
# Currently there are several temporary files created and not deleted, and hardcoded filenames -- this may be good enough for our own evaluation purposes -- can revisit
# The source_fiber_dir and target_fiber_dir assume the same number of fiber tracts with the same names (and in both tck and vtk format)
#
# Relies on scripts in the same directory: register_dwi_ants.py, convert_tck_to_vtk.py, compute_tract_distance.py

import argparse
from pathlib import Path
import subprocess
import os
import glob

import ants

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

# === Call the script to register the source and target fa images to generate the diffeormorphism ===

script_path = Path(os.path.realpath(__file__))
register_ants_script = '%s/register_dwi_ants.py' %(str(script_path.parent))

print("\n  Running ants registration...")

subprocess.run(['python', register_ants_script, str(source_path), str(source_mask_path), str(target_path), str(target_mask_path), str(output_path)], stdout=subprocess.DEVNULL)

# === Convert ants transformations to mrtrix format ===
convert_transform_script = '%s/convert_ants_transform_to_mrtrix_transform.py' %(str(script_path.parent))

forward_transform = '%s/diffeo_Composite.h5' %(str(output_path))
inverse_transform = '%s/diffeo_InverseComposite.h5' %(str(output_path))

print("\n  Converting transformations to mrtrix...")

subprocess.run(['python', convert_transform_script, '--out_name', 'mrtrix_warp_forward', str(target_mask_path), forward_transform, str(output_path)], stdout=subprocess.DEVNULL)
subprocess.run(['python', convert_transform_script, '--out_name', 'mrtrix_warp_inverse', str(target_mask_path), inverse_transform, str(output_path)], stdout=subprocess.DEVNULL)

# === Warp the fibers using mtrtrix ===  
fiber_out_path = Path('%s/warped_fibers' %(str(output_path)))

if not fiber_out_path.exists():
    os.mkdir(str(fiber_out_path))

all_source_fibers = sorted(glob.glob(str(source_fiber_path) + '/*.tck'))
all_target_fibers = sorted(glob.glob(str(target_fiber_path) + '/*.tck'))

fiber_distance_path = Path('%s/fiber_distances' %(str(output_path)))

if not fiber_distance_path.exists():
    os.mkdir(str(fiber_distance_path))

convert_to_vtk_script = '%s/convert_tck_to_vtk.py' %(str(script_path.parent)) 
fiber_distance_script = '%s/compute_tract_distance.py' %(str(script_path.parent))

print()

for i in range(0, len(all_source_fibers)):
 
    cur_fiber = all_source_fibers[i]

    warped_fiber = '%s/%s_warped.tck' %(str(fiber_out_path), Path(cur_fiber).stem)
    mrtrix_inverse = '%s/mrtrix_warp_inverse.nii.gz' %(str(output_path))

    subprocess.run(['tcktransform', cur_fiber, mrtrix_inverse, warped_fiber, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Convert the warped fibers to vtk
    subprocess.run(['python', convert_to_vtk_script, str(fiber_out_path), str(fiber_out_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # === Compute fiber tract distance ===
    cur_distance_log_path = '%s/%s_distance.txt' %(str(fiber_distance_path), Path(cur_fiber).stem)
    cur_distance_log = open(cur_distance_log_path, 'w')
    warped_fiber_vtk = '%s/%s_warped.vtk' %(str(fiber_out_path), Path(cur_fiber).stem)
    target_fiber_vtk = '%s/%s.vtk' %(str(target_fiber_path), Path(all_target_fibers[i]).stem)

    print("  Computing fiber tract distance " + str(i+1) + " of " + str(len(all_source_fibers)))

    subprocess.run(['python', fiber_distance_script, '--percent_sample_fibers', str(percent_sample_fibers), warped_fiber_vtk, target_fiber_vtk], stdout=cur_distance_log)
    cur_distance_log.close()

print()