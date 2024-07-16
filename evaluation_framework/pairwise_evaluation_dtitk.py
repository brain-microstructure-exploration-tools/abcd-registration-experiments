# Evaluates dtitk pairwise registration between a source and a target dti images

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib

from evaluation_lib import dwi_registration, pairwise_evaluation

parser = argparse.ArgumentParser(description='Performs pairwise dti registration using dtitk and evaluates the accuracy via fiber tract distance and dice overlap')

# Optional arguments
parser.add_argument('--percent_sample_fibers', default=0.1, type=float, help='randomly sample a percentage of fiber streamlines')
parser.add_argument('--num_repeats', default=1, type=int, help='the number of times to repeat the fiber tract distance measurement')
parser.add_argument('--force', action="store_true", help='force the full experiment to be run and overwrite any files already present')

# Positional arguments next
parser.add_argument('source', type=str, help='path to a source dti')
parser.add_argument('source_fiber_dir', type=str, help='directory for source fiber tracts')
parser.add_argument('source_segmentation_dir', type=str, help='directory for source binary segmentations')
parser.add_argument('target', type=str, help='path to a target dti')
parser.add_argument('target_mask', type=str, help='path to a target brain mask')
parser.add_argument('target_fiber_dir', type=str, help='directory for target fiber tracts')
parser.add_argument('target_segmentation_dir', type=str, help='directory for target binary segmentations')
parser.add_argument('output_base_dir', type=str, help='path to a base folder for saving output of registration')
parser.add_argument('exp_name', type=str, help='a name for the experiment')

args = parser.parse_args()

percent_sample_fibers = args.percent_sample_fibers
num_repeats = args.num_repeats
force_rerun = args.force

source_path = Path(os.path.abspath(args.source))
source_without_ext = str(source_path)[:str(source_path).rfind(''.join(source_path.suffixes))]
source_fiber_path = Path(os.path.abspath(args.source_fiber_dir))
source_segmentation_path = Path(os.path.abspath(args.source_segmentation_dir))

target_path = Path(os.path.abspath(args.target))
target_without_ext = str(target_path)[:str(target_path).rfind(''.join(target_path.suffixes))]
target_fiber_path = Path(os.path.abspath(args.target_fiber_dir))
target_segmentation_path = Path(os.path.abspath(args.target_segmentation_dir))

target_mask_path = Path(os.path.abspath(args.target_mask))

output_path = Path(os.path.abspath(args.output_base_dir))

if not output_path.exists():
  os.mkdir(str(output_path))

# Check source files
if not source_path.exists():
    raise FileNotFoundError(f"File {source_path} not found")

# Check target files  
if not target_path.exists():
    raise FileNotFoundError(f"File {target_path} not found")

# Check target files  
if not target_mask_path.exists():
    raise FileNotFoundError(f"File {target_mask_path} not found")

# Output for the experiment
output_base_dir = Path(os.path.abspath(args.output_base_dir))
exp_name = args.exp_name

output_path = output_base_dir / exp_name

if not output_path.exists():
    os.mkdir(str(output_path))

# This will be a directory to store everything related to the registration
registration_path = output_path / 'registration'

if not registration_path.exists():
    os.mkdir(str(registration_path))

# A json file to store information about the registration experiment
experiment_json_file = output_path / f'{exp_name}.json'

# Flag to determine if this is a new or continuing experiment
new_exp = True
experiment_dict = {}

# If an experiment file with this name already exists let's read it
if experiment_json_file.exists():

    # Opening experiment json file
    jsonf = open(experiment_json_file)
    experiment_dict = json.load(jsonf)
    # This is not a new experiment
    new_exp = False

    # Make sure this is the same experiment (the source/target are the same)
    if ((experiment_dict["source_image"] != str(source_path)) or (experiment_dict["target_image"] != str(target_path))) and (not force_rerun):
        sys.exit("It appears this experiment has already been performed with a different source and/or target. To overwrite previous experiment run with --force.")

# Let's summarize the experiment with a dictionary
# This sould be done if the user is forcing a registration rerun OR this is the first time running the experiment
if (force_rerun) or (new_exp):
    experiment_dict["driver"] = os.path.realpath(__file__)
    experiment_dict["datetime"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    experiment_dict["experiment_name"] = exp_name
    experiment_dict["source_image"] = str(source_path)
    experiment_dict["source_fibers"] = str(source_fiber_path)
    experiment_dict["source_segmentations"] = str(source_segmentation_path)
    experiment_dict["target_image"] = str(target_path)
    experiment_dict["target_mask"] = str(target_mask_path)
    experiment_dict["target_fibers"] = str(target_fiber_path)
    experiment_dict["target_segmentations"] = str(target_segmentation_path)
    experiment_dict["registration_method"] = dwi_registration.RegistrationMethods.DTITK
    experiment_dict["percent_sample_fibers"] = str(percent_sample_fibers)
    experiment_dict["num_repeats"] = str(num_repeats)

# Deformation field files
forward_diffeo_filename = registration_path / 'diffeo_forward.nii.gz'
inverse_diffeo_filename = registration_path / 'diffeo_inverse.nii.gz'

# If this is a continued experiment where the user is NOT reruning, check to make sure the deformation fields exists. If they don't exist let the user know a force is required
if (not forward_diffeo_filename.exists()) and (not new_exp and not force_rerun):
    sys.exit("Missing dtitk deformation fields. The full experiment must be rerun with --force.")

if (force_rerun) or (new_exp):

    print("\n  Running dtitk registration...")

    start_time = time.time()
    forward_diffeo = dwi_registration.register_dtitk_dti(source_path, target_path, target_mask_path, diffusivity_scale=1875)
    duration = time.time() - start_time

    experiment_dict["registration_runtime"] = '%0.2f seconds' %(duration)

    # Write the forward diffeo (we will invert the diffeo with mrtrix later)
    nib.save(forward_diffeo, str(forward_diffeo_filename))

else:

    print("\n  Using previously computed dtitk registration results...")

print("\n  Running dtitk evalutation...")


pairwise_evaluation.pairwise_evaluation_dtitk(forward_diffeo_filename, inverse_diffeo_filename, \
                                            source_fiber_path, source_segmentation_path, target_fiber_path, target_segmentation_path, \
                                            output_path, percent_sample_fibers=percent_sample_fibers, num_repeats=num_repeats, \
                                            specified_fibers=pairwise_evaluation.TESTING_FIBER_TRACTS, \
                                            specified_segmentations=pairwise_evaluation.DEFAULT_SEGMENTATIONS)

print()

# Write experiment json
json_object = json.dumps(experiment_dict, indent=4)

with open(experiment_json_file, "w") as outfile:
    outfile.write(json_object)