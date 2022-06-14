import os
import glob
import pathlib
import multiprocessing as mp
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dipy.io.image
import dipy.io
import dipy.core.gradients
import dipy.reconst.dti
import dipy.segment.mask

parser = argparse.ArgumentParser(description='Batch process diffusion weighted images: brain mask, DTI fit, and compute FA images.')
parser.add_argument('numParallel', type=int, nargs='?', default=mp.cpu_count(),
    help='number of processes to run at a time'
)
parser.add_argument('--recomputeMask', default=False, action="store_true",
    help='force computation of brain mask even if file exists'
)
parser.add_argument('--recomputeDTI', default=False, action="store_true",
    help='force computation of DTI even if file exists'
)
parser.add_argument('--recomputeFA', default=False, action="store_true",
    help='force computation of FA even if file exists'
)
args = parser.parse_args()

num_parallel = args.numParallel
recompute_mask = args.recomputeMask
recompute_dti = args.recomputeDTI
recompute_fa = args.recomputeFA

# Define source image and output locations
data_dir = '/home/ebrahim/data/abcd/DMRI_extracted'
img_dirs = glob.glob(os.path.join(data_dir,'*ABCD-MPROC-DTI*/sub-*/ses-*/dwi/'))
output_dir = './dti_fit_images/'
output_dir_dti = os.path.join(output_dir,'dti')
output_dir_fa = os.path.join(output_dir,'fa')
output_dir_fa_preview = os.path.join(output_dir,'fa_preview')
output_dir_brainmask = os.path.join(output_dir,'brainmask')
pathlib.Path(output_dir_brainmask).mkdir(exist_ok=True)
pathlib.Path(output_dir_dti).mkdir(exist_ok=True)
pathlib.Path(output_dir_fa).mkdir(exist_ok=True)
pathlib.Path(output_dir_fa_preview).mkdir(exist_ok=True)

# Using the sampled table, generate a list of dicts where each dict represents one diffusion weighted dataset
sampled_fmriresults01_df = pd.read_csv('01.0_abcd_sample/sampled_fmriresults01.csv')
sampled_fmriresults01_df['dirname'] = sampled_fmriresults01_df.derived_files.apply(lambda x : x.split('/')[-1].strip('.tgz'))
dirname_to_full_path = {img_dir.split('/')[-5]:img_dir for img_dir in img_dirs}
data = []
for (subjectkey,interview_age),df in sampled_fmriresults01_df.groupby(['subjectkey', 'interview_age']):
    paths = []
    for _,row in df.iterrows():
        if row.dirname not in dirname_to_full_path.keys():
            raise FileNotFoundError(f"Could not find a directory for fmriresults01 id {row.fmriresults01_id}")
        img_dir = dirname_to_full_path[row.dirname]
        dwi_path = glob.glob(os.path.join(img_dir, '*.nii'))[0]
        bval_path = glob.glob(os.path.join(img_dir, '*.bval'))[0]
        bvec_path = glob.glob(os.path.join(img_dir, '*.bvec'))[0]
        paths.append({
            'img_dir' : img_dir,
            'dwi_path' : dwi_path,
            'bval_path' : bval_path,
            'bvec_path' : bvec_path,
        })
    data.append({
        'paths' : paths,

        'subjectkey' : row.subjectkey,
        'interview_age' : row.interview_age,
    })

# Function to load data from one of the dictionaries listed in the object "data" defined above
def load_data(d):
    img_data_list =[]
    bvals_list = []
    bvecs_list = []
    prev_affine_transform = None

    for p in d['paths']:
        img_data, affine = dipy.io.image.load_nifti(p['dwi_path'])
        assert((prev_affine_transform is None) or (affine==prev_affine_transform).all())
        prev_affine_transform = affine
        bvals, bvecs = dipy.io.read_bvals_bvecs(p['bval_path'], p['bvec_path'])
        img_data_list.append(img_data)
        bvals_list.append(bvals)
        bvecs_list.append(bvecs)
        bvals = np.concatenate(bvals_list)
    img_data = np.concatenate(img_data_list, axis=-1)
    bvecs = np.concatenate(bvecs_list, axis=0)
    gtab = dipy.core.gradients.gradient_table(bvals, bvecs)
    return img_data, affine, gtab

def process_data_item(d):

    output_file_basename = f"{d['subjectkey']}-{d['interview_age']}"

    print(f"Processing {output_file_basename}...")

    # Load the diffusion weighted dataset
    img_data, affine, gtab = load_data(d)

    # Use median Otsu method for masking to the brain region
    mask_path = os.path.join(output_dir_brainmask,f'{output_file_basename}.nii.gz')
    if recompute_mask or not os.path.exists(mask_path):
        img_data_masked, mask = dipy.segment.mask.median_otsu(img_data, vol_idx = range(img_data.shape[-1]))
        dipy.io.image.save_nifti(mask_path, mask.astype(np.float32), affine)
    else:
        mask_loaded, mask_loaded_affine = dipy.io.image.load_nifti(mask_path)
        assert((mask_loaded_affine == affine).all())
        img_data_masked = img_data * np.repeat(mask_loaded[:,:,:,np.newaxis], img_data.shape[-1], axis=-1)

    # Fit tensor model
    lt_path = os.path.join(output_dir_dti,f'{output_file_basename}.nii.gz')
    if recompute_dti or not os.path.exists(lt_path):
        tensor_model = dipy.reconst.dti.TensorModel(gtab)
        tensor_fit = tensor_model.fit(img_data_masked)
        lt = tensor_fit.lower_triangular()
        dipy.io.image.save_nifti(lt_path, lt, affine)
        eigvals = tensor_fit.evals
    else:
        lt, lt_affine = dipy.io.image.load_nifti(lt_path)
        assert((lt_affine == affine).all())
        eig = dipy.reconst.dti.eig_from_lo_tri(lt) # has eigenvals and eigenvecs
        eigvals = eig[:,:,:,:3] # take only the eigenvals


    # Compute the FA image
    fa_path = os.path.join(output_dir_fa,f'{output_file_basename}.nii.gz')
    if recompute_fa or not os.path.exists(fa_path):
        fa = dipy.reconst.dti.fractional_anisotropy(eigvals)
        dipy.io.image.save_nifti(fa_path, fa, affine)
    else:
        fa, fa_affine = dipy.io.image.load_nifti(fa_path)
        assert((fa_affine == affine).all())

    # Point out if there are NaNs
    num_nans = np.isnan(fa).sum()
    if num_nans > 0:
        print(f"Got {num_nans} NaNs in the FA image for subject {d['subjectkey']}, age {d['interview_age']}")

    # Save a preview image
    fig,axs = plt.subplots(1,2,figsize=(20,10))
    axs[0].imshow(fa[62,:,:].T, origin='lower', cmap='gray')
    axs[1].imshow(fa[:,:,80].T, origin='lower', cmap='gray')
    plt.savefig(os.path.join(output_dir_fa_preview, f'{output_file_basename}.png'))


print(f"There are {len(data)} diffusion weighted datasets to process; running {num_parallel} processes in parallel.")
with mp.Pool(num_parallel) as p:
    p.map(process_data_item,data)