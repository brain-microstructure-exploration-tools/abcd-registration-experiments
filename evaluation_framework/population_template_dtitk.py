# Estimation of a population template by pairwise ddwi registration using dti-tk (requires dti-tk installed and setup per instructions)
# https://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Install

# Assumes images are rigidly aligned to start

import argparse
import glob
import os
import subprocess
from pathlib import Path

import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# === Parse args ===

parser = argparse.ArgumentParser(description='performs non-linear template estimation for a folder of dwi images')

parser.add_argument('--iters', default=5, type=int, help='number of iterations for template estimation')

# Positional arguments
parser.add_argument('data_dir', type=str, help='folder with dwi images, bval, bvec, and brain masks with subject prefixes')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output of template building')

args = parser.parse_args()

num_iters = args.iters

data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)

if not data_dir.exists():
  raise FileNotFoundError(f"Path {data_dir} not found")

# If the output directory doesn't exist let's create it
if not output_dir.exists():
  os.mkdir(str(output_dir))  
  
# Also create a tmp directory to store intermediate outputs
tmp_dir = Path(str(output_dir) + '/tmp')
if not tmp_dir.exists():
  os.mkdir(str(tmp_dir))  
   
# Let's read the some data
dwi_list = sorted(glob.glob(str(data_dir) + '/*dwi.nii.gz'))
mask_list = sorted(glob.glob(str(data_dir) + '/*mask*'))
bval_list = sorted(glob.glob(str(data_dir) + '/*.bval'))
bvec_list = sorted(glob.glob(str(data_dir) + '/*.bvec'))

# List of subjects needed to compute the mean
subjects_strs = []
mask_strs = []

print('\n  Converting dwi into dti...')

# Compute DTI volumes
for i in range(0, len(dwi_list)):

  print('    Working on subject ' + str(i+1) + ' of ' + str(len(dwi_list)) + "...")

  dwi_im_path = dwi_list[i]
  bval_path = bval_list[i]
  bvec_path = bvec_list[i]
  mask_path = mask_list[i]
  
  cur_data, cur_affine, cur_img = load_nifti(dwi_im_path, return_img=True)
  cur_bvals, cur_bvecs = read_bvals_bvecs(bval_path, bvec_path)

  cur_gtab = gradient_table(cur_bvals, cur_bvecs)

  cur_mask_data, cur_mask_affine, cur_mask_img = load_nifti(mask_path, return_img=True)

  cur_tenmodel = dti.TensorModel(cur_gtab)
  cur_tenfit = cur_tenmodel.fit(cur_data, mask=cur_mask_data)

  cur_dti_lotri = cur_tenfit.lower_triangular()

  cur_out_dti_filename = '%s/tmp/subj_%0.4d_dti.nii.gz' %(output_dir, i)
  save_nifti(cur_out_dti_filename, cur_dti_lotri, cur_affine, cur_img.header)
  
  cur_out_mask_filename = '%s/tmp/subj_%0.4d_mask.nii.gz' %(output_dir, i)
  save_nifti(cur_out_mask_filename, cur_mask_data, cur_mask_affine, cur_mask_img.header)
  
  # Preprocess the dti for dti-tk
  subprocess.run(['TVtool', '-in', cur_out_dti_filename, '-scale', '1500', '-out', cur_out_dti_filename], stdout=subprocess.DEVNULL)
  subprocess.run(['TVAdjustVoxelspace', '-in', cur_out_dti_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
  subprocess.run(['SVAdjustVoxelspace', '-in', cur_out_mask_filename, '-origin', '0', '0', '0'], stdout=subprocess.DEVNULL)
  subprocess.run(['TVResample', '-in', cur_out_dti_filename, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)
  subprocess.run(['SVResample', '-in', cur_out_mask_filename, '-align', 'origin', '-size', '128', '128', '128', '-vsize', '1.86', '1.86', '1.86'], stdout=subprocess.DEVNULL)

  # Add to our list of subjects
  subjects_strs.append(cur_out_dti_filename + '\n')
  mask_strs.append(cur_out_mask_filename + '\n')
  
list_of_subjects_path = '%s/subjects.txt' %(str(tmp_dir))
list_of_subjects_file = open(list_of_subjects_path,'w')  
list_of_subjects_file.writelines(subjects_strs)
list_of_subjects_file.close()

list_of_masks_path = '%s/masks.txt' %(str(tmp_dir))
list_of_masks_file = open(list_of_masks_path,'w')  
list_of_masks_file.writelines(mask_strs)
list_of_masks_file.close()

# Compute the inital template as the average of the dti
init_template = '%s/initial_template.nii.gz' %(str(tmp_dir))
subprocess.run(['TVMean', '-in', list_of_subjects_path, '-out', init_template, '-type', 'ORIGINAL', '-interp', 'LEI'], stdout=subprocess.DEVNULL)
# Mean mask?
init_mask = '%s/initial_template_mask.nii.gz'  %(str(tmp_dir))
subprocess.run(['SVMean', '-in', list_of_masks_path, '-outMean', init_mask], stdout=subprocess.DEVNULL)

# === Main loop for template buildling ===

cur_template = init_template
cur_template_mask = init_mask
cur_dir = str(tmp_dir)

print()

for cur_iter in range(0, num_iters):

  print('  Working on template estimation iteration ' + str(cur_iter+1) + ' of ' + str(num_iters) + '...')

  cur_iter_dir = Path('%s/iter_%0.4d' %(str(tmp_dir), cur_iter+1))
  
  if not cur_iter_dir.exists():
    os.mkdir(str(cur_iter_dir))
    
  processes = []
  warped_dtis = []
  warped_masks = []
  
  for i in range(0, len(dwi_list)):
  
    subj_folder = Path('%s/subj_%0.4d' %(str(cur_iter_dir), i))
    
    if not subj_folder.exists():
      os.mkdir(str(subj_folder))
      
    # Keep track of each warped dti this iteration to update the template
    warped_dti_path = '%s/subj_%0.4d_dti_aff_diffeo.nii.gz' %(subj_folder, i)
    warped_dtis.append(warped_dti_path + '\n')
    # Ideally we need to upate the template mask each iteration but that is TODO
      
    cur_source = '%s/subj_%0.4d_dti.nii.gz' %(cur_dir, i)
    cur_source_mask = '%s/subj_%0.4d_mask.nii.gz' %(cur_dir, i)
      
    p = subprocess.Popen(['python', 'register_dti_dtitk.py', '--scale', '1', '--iters', '6', cur_source, cur_source_mask, cur_template, cur_template_mask, subj_folder], stdout=subprocess.DEVNULL)
    
    processes.append(p)
  
  # Launch parallel processes
  for p in processes:
    p.wait()
    
  list_of_warped_dti_path = '%s/warped_dti.txt' %(str(cur_iter_dir))
  list_of_warped_dti_file = open(list_of_warped_dti_path,'w')  
  list_of_warped_dti_file.writelines(warped_dtis)
  list_of_warped_dti_file.close()
  
  cur_template = '%s/template_iter_%0.4d.nii.gz' %(str(cur_iter_dir), cur_iter+1)
  subprocess.run(['TVMean', '-in', list_of_warped_dti_path, '-out', cur_template, '-type', 'ORIGINAL', '-interp', 'LEI'], stdout=subprocess.DEVNULL)
