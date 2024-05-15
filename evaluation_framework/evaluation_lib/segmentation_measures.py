import os
from pathlib import Path

import numpy as np
import vtk

import nibabel as nib

def dice_overlap(source_seg_path: Path, target_seg_path: Path) -> float:
    """
    Computes the dice overlap between two binary segmentations (https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient)
    This method assumes a binary label map with value 0 for background and 1 for object

    :param source_seg_path: path to a binary segmentation in nii.gz format
    :param target_seg_path: path to a binary segmentation in nii.gz format
    :return: the dice overlap, a value in the range between 0 and 1
    """
    
    source_im_nib = nib.load(str(source_seg_path))
    source_im = np.array(source_im_nib.dataobj)

    target_im_nib = nib.load(str(target_seg_path))
    target_im = np.array(target_im_nib.dataobj)

    return np.sum(source_im[target_im==1])*2.0 / (np.sum(source_im) + np.sum(target_im))