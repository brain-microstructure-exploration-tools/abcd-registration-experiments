This repository contains various scripts and notebooks related to my exploration of [dMRI](https://en.wikipedia.org/wiki/Diffusion_MRI) [image registration](https://en.wikipedia.org/wiki/Image_registration) for the [ABCD Study](https://abcdstudy.org/) dataset.

# Deformable registration tool for FA datasets

This tool uses a deep learning based nonlinear registration model to do pairwise registration of FA images.

Given a dataset of FA images, the "most typical" image can be identified by registering all possible pairs of images and computing the mean squared displacement needed to achieve alignment. The "most typical" image is the one for which the least displacement is needed in order to align all other images to it.

All images can then be aligned to the "most typical" image, and the aligned images can be saved.

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