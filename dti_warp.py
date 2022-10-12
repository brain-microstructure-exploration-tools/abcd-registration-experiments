
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

  def __init__(self,
    device='cpu',
    tensor_transform_type = TensorTransformType.FINITE_STRAIN,
    polar_decomposition_mode = 'svd',
  ) -> None:
    """
      Args:
        device: cpu, cuda, etc.
        tensor_transform_type: Method used to rotate diffusion tensors
        polar_decomposition_mode: Algorithm to use for polar decomposition.
          Only applies when tensor_transform_type is finite strain method.
          Can be "svd" to use torch.linalg.svd (gradients are sometimes numerically unstable for this),
          or can be an integer to indication doing a Newton approximation with the given number of iterations.
    """
    super().__init__()

    # spatial resampling without the tensor transform
    self.warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
    self.deriv_ddf = DerivativeOfDDF(device=device)
    self.tensor_transform_type = tensor_transform_type

    if polar_decomposition_mode=='svd':
      self.use_svd = True
    else:
      self.use_svd = False
      self.polar_decomp_iterations = int(polar_decomposition_mode)

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

      # Get the orthogonal component of the jacobian, in the sense of polar decomposition.
      if self.use_svd:
        U, _, Vh = torch.linalg.svd(J_mat_nospatial, full_matrices=False)
        Jrot_mat_nospatial = torch.matmul(U, Vh)
      else:
        U = J_mat_nospatial
        for _ in range(self.polar_decomp_iterations):
          zeta = torch.linalg.det(U).abs()**(-1/3)
          zU = zeta.view(-1,1,1) * U
          U = 0.5 * ( zU + torch.linalg.inv(zU.permute(0,2,1)) )
        Jrot_mat_nospatial = U

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

class MseLossDTI(nn.Module):
  """Compare two DTIs and return the spatially averaged squared distance between diffusion tensors,
  where two diffusion tensors D1 and D2 are compared using the Frobenius norm, which is equivalent
  to tr((D1-D2)^2) because they are symmetric matrices.
  """
  def __init__(self, device='cpu') -> None:
    super().__init__()

    self.symmetrization_multiplier = torch.tensor([1,2,1,2,2,1]).view((1,-1,1,1,1)).to(device)

  def forward(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Input two batches of DTIs for comparison, each shaped (B,6,H,W,D).
    The channel dimension is interpreted to be the lower triangular entries of
    the diffusion tensors, in the ordering used by dipy: (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)"""
    return (((b1-b2)**2) * self.symmetrization_multiplier).sum(dim=1).mean()
