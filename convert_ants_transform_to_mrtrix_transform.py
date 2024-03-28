import argparse
from pathlib import Path
import subprocess
import os

import ants

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
  
identity_warp = '%s/identity_warp[].nii.gz' %(str(output_dir))

# Create identity warp
subprocess.run(['warpinit', target_image, identity_warp, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Transform the idenity warp using ants

ants_target_im = ants.image_read(target_image)

# Transform the 3 dimensions
for d in range(0, 3):

    cur_identity_warp = '%s/identity_warp%d.nii.gz' %(output_dir, d)

    cur_ants_source_im = ants.image_read(cur_identity_warp)
    
    cur_warped_image = ants.apply_transforms(fixed=ants_target_im, moving=cur_ants_source_im, transformlist=ants_transform, defaultvalue=2147483647)

    out_image_filename = '%s/mrtrix_warp%d.nii.gz' %(output_dir, d)
    ants.image_write(cur_warped_image, out_image_filename)

# Fix warp
corrected_warp = '%s/%s.mif' %(output_dir, out_name)
warp_filename = '%s/mrtrix_warp[].nii.gz' %(output_dir)
subprocess.run(['warpcorrect', warp_filename, corrected_warp, '-marker', '2147483647', '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Perhaps we should clean up the temporary files that are no longer needed?



