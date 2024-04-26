import argparse
from pathlib import Path
import subprocess
import os

import nibabel as nib
import ants

import transformation_utils

parser = argparse.ArgumentParser(description='Converts an ants transform to a transform compatible with mrtrix')

parser.add_argument('--out_name', default='final_mrtrix_transform', type=str, help='optional name for the output transformation file')

parser.add_argument('target_image', type=str, help='defines the spatial characteristics of the warp (spacing/size/etc.)')
parser.add_argument('ants_transform', type=str, help='path to an ants transform file')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output')

args = parser.parse_args()

target_image = args.target_image
ants_transform = args.ants_transform
output_dir = Path(args.output_dir)

out_name = args.out_name

# Create the output directory if it doesn't exist
if not output_dir.exists():
  os.mkdir(str(output_dir))  

mrtrix_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_image, ants_transform)

out_warp_filename = '%s/%s.nii.gz' %(str(output_dir), out_name)
nib.save(mrtrix_warp, out_warp_filename)



