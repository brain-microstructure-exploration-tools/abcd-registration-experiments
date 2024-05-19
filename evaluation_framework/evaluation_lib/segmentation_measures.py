from pathlib import Path

import numpy as np
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

def volumetric_similarity(source_seg_path: Path, target_seg_path: Path) -> float:
    """
    Computes the volumetric similarity between two binary segmentations defined by eqn (3) https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x
    This method assumes a binary label map with value 0 for background and 1 for object

    :param source_seg_path: path to a binary segmentation in nii.gz format
    :param target_seg_path: path to a binary segmentation in nii.gz format
    :return: the volumetric similarity, a value in the range between 0 and 1
    """

    source_im_nib = nib.load(str(source_seg_path))
    source_im = np.array(source_im_nib.dataobj)

    target_im_nib = nib.load(str(target_seg_path))
    target_im = np.array(target_im_nib.dataobj)

    source_volume = (source_im == 0).sum()
    target_volume = (target_im == 0).sum()

    return 1 - abs(source_volume - target_volume) / (source_volume + target_volume)