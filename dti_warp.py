
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_derivatives import DerivativeOfDDF
from enum import Enum


# Name some operations to make it easier to interpret the steps below
dipy2torch_lotri_batch = lambda t : t.permute(0,4,1,2,3)
torch2dipy_lotri_batch = lambda t : t.permute(0,2,3,4,1)
dipy2torch_mat_batch = lambda t : t.permute(0,4,5,1,2,3)
torch2dipy_mat_batch = lambda t : t.permute(0,3,4,5,1,2)
torch_lotri2mat_batch = lambda t : t[:,[[0,1,3],[1,2,4],[3,4,5]],:,:,:]
torch_mat2lotri_batch = lambda t : t[:,[0, 1, 1, 2, 2, 2],[0, 0, 1, 0, 1, 2],:,:,:]
torch_mat_batch_absorbspatial = lambda t : t.permute((0,3,4,5,1,2)).reshape((-1,3,3))
torch_mat_batch_expandspatial = lambda t,b,h,w,d : t.reshape(b,h,w,d,3,3).permute((0,4,5,1,2,3))

class TensorTransformType(Enum):
  NONE = 0
  FINITE_STRAIN = 1

class WarpDTI(nn.Module):
    r"""Apply a DDF to warp a diffusion tensor image.

    Uses the finite strain method for diffusion tensor transformation.
    See https://pubmed.ncbi.nlm.nih.gov/11700739/

    Use ordinary linear interpolation on diffusion tensors.
    This is an area for improvement; see https://pubmed.ncbi.nlm.nih.gov/15690523/

    Args:
        dti: A batch of diffusion tensor images in lower triangular form.
        ddf: A displacement vector field.

    Returns:
        A warped and transformed batch of diffusion tensor images in lower triangular form.

    Shape:
        - Input: :math:`(B, 6, H, W, D)`, :math:`(B, 3, H, W, D)`
        - Output: :math:`(B, 6, H, W, D)`
    """

    def __init__(self, device='cpu', tensor_transform_type = TensorTransformType.FINITE_STRAIN) -> None:
        super().__init__()

        # spatial resampling without the tensor transform
        self.warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
        self.deriv_ddf = DerivativeOfDDF(device=device)
        self.tensor_transform_type = tensor_transform_type

    def forward(self, dti: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
        if (len(dti.shape)!=5 or dti.shape[1]!=6):
          raise ValueError(f"Expected dti to have shape (B, 6, H, W, D), but got {tuple(dti.shape)}")
        b,_,h,w,d = dti.shape
        if (len(ddf.shape)!=5 or ddf.shape[1]!=3):
          raise ValueError(f"Expected ddf to have shape (B, 3, H, W, D), but got {tuple(ddf.shape)}")
        if (ddf.shape[2:] != dti.shape[2:]):
          raise ValueError(f"Expected dti to have same spatial dimensions as ddf, but got {tuple(dti.shape[2:])} and {tuple(ddf.shape[2:])}")

        # Warp the DTI, spatially moving tensors but not transforming the tensors yet
        dti_warped_lotri = self.warp(dti, ddf)

        if self.tensor_transform_type == TensorTransformType.FINITE_STRAIN:
          # Convert from lower-triangular form to the 3x3 symmetric matrices
          dti_warped_mat = torch_lotri2mat_batch(dti_warped_lotri)

          # Move the spatial dimensions into the batch dimension, so we can do some linear algebra
          # at the level of the individual diffusion tensors
          dti_warped_mat_nospatial = torch_mat_batch_absorbspatial(dti_warped_mat)

          # Compute the jacobian matrix of the DDF and shape it as 3x3 matrices at each point
          J_mat = self.deriv_ddf(ddf).reshape(b,3,3,h,w,d)

          # Move the spatial dimensions into the batch dimension
          J_mat_nospatial = torch_mat_batch_absorbspatial(J_mat)

          # Get SVD of jacobian and derive from it the orthogonal component of the jacobian,
          # in the sense of polar decomposition.
          U, _, Vh = torch.linalg.svd(J_mat_nospatial, full_matrices=False)
          Jrot_mat_nospatial = torch.matmul(U, Vh)

          # Transform tensors using the tensor transformation law, but using only the rotational component Jrot of J
          # This is the so-called finite strain method.
          dti_warped_transformed_mat_nospatial = torch.matmul(
            Jrot_mat_nospatial.permute(0,2,1),
            torch.matmul(
              dti_warped_mat_nospatial,
              Jrot_mat_nospatial,
            )
          )

          # Move the spatial dimensions back out of the batch dimension
          dti_warped_transformed_mat = torch_mat_batch_expandspatial(dti_warped_transformed_mat_nospatial, b,h,w,d)

          # Convert back to lower triangular form
          dti_warped_transformed_lotri = torch_mat2lotri_batch(dti_warped_transformed_mat)

        elif self.tensor_transform_type == TensorTransformType.NONE:
          dti_warped_transformed_lotri = dti_warped_lotri

        else:
          raise NotImplementedError('Tensor transform method not recognized.')

        return dti_warped_transformed_lotri

