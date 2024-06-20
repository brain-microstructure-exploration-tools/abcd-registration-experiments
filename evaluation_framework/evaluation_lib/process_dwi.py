import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti


def reconstruct_dti(dwi_path, bval_path, bvec_path, mask_path):

    if not dwi_path.exists() or not bval_path.exists() or not bvec_path.exists():
        raise FileNotFoundError(f"File {dwi_path} not found (or missing corresponding bval/bvec)")
  
    if not mask_path.exists():
        raise FileNotFoundError(f"File {mask_path} not found")
    
    dwi_data, dwi_affine, dwi_img = load_nifti(str(dwi_path), return_img=True)
    dwi_bvals, dwi_bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))

    dwi_gtab = gradient_table(dwi_bvals, dwi_bvecs)

    dwi_mask_data, dwi_mask_affine, dwi_mask_img = load_nifti(mask_path, return_img=True)

    tenmodel = dti.TensorModel(dwi_gtab)
    tenfit = tenmodel.fit(dwi_data, mask=dwi_mask_data)

    dti_lotri = tenfit.lower_triangular()
    
    dwi_info = {}
    dwi_info['affine'] = dwi_affine
    dwi_info['header'] = dwi_img.header

    return dwi_info, dti_lotri, tenfit.fa
