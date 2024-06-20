import argparse
import os

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description='Removes axes of length one from a nifti image')

parser.add_argument('im_in', type=str, help='path to input nifti image')
parser.add_argument('im_out', type=str, help='path to output nifti image')

args = parser.parse_args()

filename_in = os.path.abspath(args.im_in)
filename_out = os.path.abspath(args.im_out)

img = nib.load(filename_in)
data = img.get_fdata()

reshaped_data = np.squeeze(data)

# Preserve affine and header information
affine = img.affine
header = img.header

# Create a new nifti image with reshaped data
new_img = nib.Nifti1Image(reshaped_data, affine=affine, header=header)

# Save the new nifti file (modify filename to avoid overwrite)
nib.save(new_img, filename_out)
