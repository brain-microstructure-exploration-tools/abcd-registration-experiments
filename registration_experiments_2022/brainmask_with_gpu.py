from medianblur_with_gpu import MedianBlur
from skimage.filters import threshold_otsu
from dipy.segment.mask import applymask
import numpy as np
import torch

def median_otsu_gpu(input_volume, vol_idx=None, median_radius=2, numpass=10):
    """
    A version of DIPY's median_otsu filter that runs on GPU.
    """
    if len(input_volume.shape) == 4:
        if vol_idx is not None:
            b0vol = np.mean(input_volume[..., tuple(vol_idx)], axis=3)
        else:
            raise ValueError("For 4D images, must provide vol_idx input")
    else:
        b0vol = input_volume

    kernel_size = (2*median_radius+1,) * 3
    median_blur = MedianBlur(kernel_size, device='cuda')

    mask = median_blur(
      torch.tensor(b0vol).to('cuda').unsqueeze(0).unsqueeze(0),
      numpass=numpass
    )[0,0].cpu().numpy()
    thresh = threshold_otsu(mask)
    mask = mask > thresh

    maskedvolume = applymask(input_volume, mask)
    return maskedvolume, mask