import argparse
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description='Performs tractography on a single subject')

parser.add_argument('fod_im', type=str, help='input image of spherical harmonic coefficients')
parser.add_argument('out_dir', type=str, help='path to desired output directory')

args = parser.parse_args()

fod_im_path = Path(os.path.abspath(args.fod_im))
out_dir = Path(os.path.abspath(args.out_dir))

if not out_dir.exists():
  os.mkdir(str(out_dir))

print('\n  Computing peaks...')

# First we find the peaks from the SH coefficients
peaks_image = '%s/peaks.nii.gz' %(str(out_dir))
print(peaks_image)
subprocess.run(['sh2peaks', fod_im_path, peaks_image], stdout=subprocess.DEVNULL)

print('\n  Computing bundle segmentations...')

# Next we run a series of commands using TractSeg to estimate tractography
subprocess.run(['TractSeg', '-i', peaks_image, '--output_type', 'tract_segmentation'], stdout=subprocess.DEVNULL)

print('\n  Computing ending segmentations...')

subprocess.run(['TractSeg', '-i', peaks_image, '--output_type', 'endings_segmentation'], stdout=subprocess.DEVNULL)

print('\n  Computing TOMS...')

subprocess.run(['TractSeg', '-i', peaks_image, '--output_type', 'TOM'], stdout=subprocess.DEVNULL)

print('\n  Running tractography...\n')

subprocess.run(['Tracking', '-i', peaks_image], stdout=subprocess.DEVNULL)
