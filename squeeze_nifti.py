"""
  Reads a nifti file, removes an axis with the specified dimension, 
  and saves the reshaped data back to a new nifti file.

  Args:
      filename: Path to the nifti file.
  """
import nibabel as nib
import argparse
from pathlib import Path
import os
import numpy as np

parser = argparse.ArgumentParser(description='performs non-linear registration between a source and target dwi')

# Positional arguments next
parser.add_argument('img', type=str, help='path to an nifti image')

#def reshape_nifti(filename, axis_to_remove):

args = parser.parse_args()

filename = os.path.abspath(args.img)
img = nib.load(filename)
data = img.get_fdata()

reshaped_data = np.squeeze(data)

# Preserve affine and header information
affine = img.affine
header = img.header

# Create a new nifti image with reshaped data
new_img = nib.Nifti1Image(reshaped_data, affine=affine, header=header)

# Save the new nifti file (modify filename to avoid overwrite)
nib.save(new_img, filename + "_reshaped.nii.gz")
