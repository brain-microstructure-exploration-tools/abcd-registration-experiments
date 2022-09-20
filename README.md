This repository contains various scripts and notebooks related to my exploration of [dMRI](https://en.wikipedia.org/wiki/Diffusion_MRI) [image registration](https://en.wikipedia.org/wiki/Image_registration) for the [ABCD Study](https://abcdstudy.org/) dataset.

# Deformable registration tool for FA datasets

This tool uses a deep learning based nonlinear registration model to do pairwise registration of FA images.

Given a dataset of FA images, the "most typical" image can be identified by registering all possible pairs of images and computing the mean squared displacement needed to achieve alignment. The "most typical" image is the one for which the least displacement is needed in order to align all other images to it.

All images can then be aligned to the "most typical" image, and the aligned images can be saved.

Notes:
- The deformable registration model is currently experimental. It's not well-tested and it hasn't been properly compared to other existing algorithms.
- FA Images need to already be affine-aligned before using this tool. A future version of the tool may relax this requirement and include the affine alignment in the predicted deformation. This is not a limitation when working with minimally processed ABCD images, since they are already affine-aligned.
- The conept of registering all images to the "most typical" image is drawn from the TBSS pipeline, developed [here](https://doi.org/10.1016/j.neuroimage.2006.02.024). This is helpful when dealing with a non-adult population as it can be difficult to find a suitable atlas. It may also improve the potential for alignment quality, because subject images are typically much sharper than a template that is formed by averaging. Finally, and most pratically, we adopt the approach here because it requires only solving the problem of _pairwise_ registration of subject images. More effort is needed to approach the problems of template construction and alignment to a template.

## Install required dependencies

This section needs to be improved. For now:
- you need a python environment with MONAI and numpy. install MONAI by first correctly installing torch.
- if you have CUDA then it will be used

## Get trained model weights

Download [this](https://drive.google.com/uc?id=11bLdPt2Yk3_nfwNiDOeZhycV8y11NRL5) to `fa_deformable_registration_models/`

Or:
```bash
pip install gdown
gdown 'https://drive.google.com/uc?id=11bLdPt2Yk3_nfwNiDOeZhycV8y11NRL5' -O fa_deformable_registration_models/
```

## Usage

```
python deformable_reg_fa_dataset.py --help
```

## Example

Suppose you have a directory `./dti_fit_images/fa` of FA images saved as `*.nii.gz` files.

Determine the "most typical" image:
```bash
python deformable_reg_fa_dataset.py dti_fit_images/fa --target dti_fit_images/fa/NDAR_INVPRNEYMH1-108.nii.gz --save-transformed-images
```

Say the most typical image turns out to be `subject123.nii.gz`. Now align all images to it and save the aligned images:
```bash
python deformable_reg_fa_dataset.py dti_fit_images/fa --target dti_fit_images/fa/subject123.nii.gz --save-transformed-images
```