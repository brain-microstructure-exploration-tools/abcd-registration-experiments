
from pickle import FROZENSET
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

class PolarDecompositionMode(Enum):
  SVD = 0
  NEWTON = 1
  HALLEY = 2
  HALLEY_DYNAMIC_WEIGHTS = 3 # see https://doi.org/10.1137/090774999

class NewtonIterationScaleFactor(Enum):
  NONE = 0
  DET = 1
  FROBENIUS = 2

def newton_iterate_none(X, N):
  U = X
  for _ in range(N):
    U = 0.5 * ( U + torch.linalg.inv(U.permute(0,2,1)) )
  return U

def newton_iterate_det(X, N):
  U = X
  for _ in range(N):
    zeta = torch.linalg.det(U).abs()**(-1/3)
    zU = zeta.view(-1,1,1) * U
    U = 0.5 * ( zU + torch.linalg.inv(zU.permute(0,2,1)) )
  return U

def newton_iterate_frobenius(X, N):
  U = X
  for _ in range(N):
    Uinv = torch.linalg.inv(U)
    zeta = torch.sqrt((Uinv**2).sum(dim=[1,2]) / (U**2).sum(dim=[1,2]))
    zU = zeta.view(-1,1,1) * U
    zUinv = (1/zeta).view(-1,1,1) * Uinv
    U = 0.5 * ( zU + zUinv.permute(0,2,1) )
  return U

newton_iterate = {
  NewtonIterationScaleFactor.NONE : newton_iterate_none,
  NewtonIterationScaleFactor.DET : newton_iterate_det,
  NewtonIterationScaleFactor.FROBENIUS : newton_iterate_frobenius,
}

def halley_iterate(X, N, id_stack = None):
  """Use Halley iteration to approximate the orthogonal component of the polar decomposition of X.
  This uses a QR-decomposition approach so that it can be free of matrix inversion.
  Args:
    X: batch of matrices, shape (B,3,3)
    N: number of halley iterations to compute
    id_stack: a stack of identity matrices, shaped (B,3,3), on the same device as X
  """
  if id_stack is None:
    id_stack = torch.repeat_interleave(torch.eye(3).unsqueeze(0),X.shape[0],dim=0).to(X)
  for _ in range(N):
    Q,_ = torch.linalg.qr( torch.cat([ X * (3**(1/2)) , id_stack], dim=1) )
    Q1 = Q[:,:3,:]
    Q2 = Q[:,3:,:]
    X = (1/3) * X + (3**(-1/2))*(3-1/3)*torch.matmul(Q1, Q2.permute(0,2,1))
  return X

def halley_iterate_QDWH(X, N, id_stack = None):
  """Use dynamically-weighted Halley iteration to approximate the orthogonal component
  of the polar decomposition of X.
  See https://doi.org/10.1137/090774999
  This uses a QR-decomposition approach so that it can be free of matrix inversion.
  Args:
    X: batch of matrices, shape (B,3,3)
    N: number of halley iterations to compute
    id_stack: a stack of identity matrices, shaped (B,3,3), on the same device as X
  """
  if id_stack is None:
    id_stack = torch.repeat_interleave(torch.eye(3).unsqueeze(0),X.shape[0],dim=0).to(X)
  svdvals = torch.linalg.svdvals(X)
  alpha = svdvals[:,0].view(-1,1,1)
  beta  = svdvals[:,2].view(-1,1,1)
  X = X / alpha.view(-1,1,1)
  l = beta/alpha
  for _ in range(N):
    d = ((4*(1-l**2))/(l**4))**(1/3)
    a = (1+d)**(1/2) + 0.5*(8-4*d+(8*(2-l**2))/(l**2*(1+d)**(1/2)))**(1/2)
    b = (a-1)**2 / 4
    c = a + b - 1
    Q,_ = torch.linalg.qr( torch.cat([ X * (c**(1/2)) , id_stack], dim=1) )
    Q1 = Q[:,:3,:]
    Q2 = Q[:,3:,:]
    X = (b/c) * X + (c**(-1/2))*(a-b/c)*torch.matmul(Q1, Q2.permute(0,2,1))
    l = torch.clamp(l*(a+b*(l**2)) / (1+c*(l**2)), max=1.0) # everything breaks if l slightly exceeds 1.0 from numerical error
  return X



class WarpDTI(nn.Module):
  r"""Apply a DDF to warp a diffusion tensor image.

  Uses the finite strain method for diffusion tensor transformation.
  See https://pubmed.ncbi.nlm.nih.gov/11700739/

  Use ordinary linear interpolation on diffusion tensors.
  This is an area for improvement; see https://pubmed.ncbi.nlm.nih.gov/15690523/

  Args:
      dti: A batch of diffusion tensor images in lower triangular form.
      ddf: A displacement vector field.
      mode: The warp interpolation mode. Use bilinear to get gradients; consider using nearest for inference.
        Note that bilinear actually means trilinear, and note that it's not really a natural way to interpolate
        over pos def symm bilinear forms like diffusion tensors.

  Returns:
      A warped and transformed batch of diffusion tensor images in lower triangular form.

  Shape:
      - Input: :math:`(B, 6, H, W, D)`, :math:`(B, 3, H, W, D)`
      - Output: :math:`(B, 6, H, W, D)`
  """

  def __init__(self,
    device='cpu',
    mode='bilinear',
    tensor_transform_type = TensorTransformType.FINITE_STRAIN,
    polar_decomposition_mode = PolarDecompositionMode.SVD,
    num_iterations : int = None,
    newton_iteration_scale_factor : NewtonIterationScaleFactor = None,
  ) -> None:
    """
      Args:
        device: cpu, cuda, etc.
        tensor_transform_type: Method used to rotate diffusion tensors
        polar_decomposition_mode: Algorithm to use for polar decomposition.
          Only applies when tensor_transform_type is finite strain method.
          Can use torch.linalg.svd (gradients are sometimes numerically unstable for this).
          Can do Newton iteration. In this case see the argument num_iterations.
          Can do Halley iteration, with standard weights or with dynamic weights. This is inverse-free.
        num_iterations: an integer indicating the number of iterations to carry out in an an approximation
          technique to computing the polar decomposition.
          Only applies when tensor_transform_type is finite strain method and polar_decomposition_mode is newton or halley.
        newton_iteration_scale_factor: what method to use for the scale factor for accelerating convergence of the newton method.
          Only applies when tensor_transform_type is finite strain method and polar_decomposition_mode is newton
    """
    super().__init__()

    # spatial resampling without the tensor transform
    self.warp = monai.networks.blocks.Warp(mode=mode, padding_mode="border")
    self.deriv_ddf = DerivativeOfDDF(device=device)

    self.device= device
    self.tensor_transform_type = tensor_transform_type
    self.polar_decomposition_mode = polar_decomposition_mode
    self.num_iterations = num_iterations
    self.newton_iteration_scale_factor = newton_iteration_scale_factor

    # helps to cache id_stack if needed, which is used for some of the techniques
    self.last_id_stack_batch_size = None

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
      if self.tensor_transform_type == TensorTransformType.FINITE_STRAIN:
        if self.polar_decomposition_mode == PolarDecompositionMode.SVD:
          U, _, Vh = torch.linalg.svd(J_mat_nospatial, full_matrices=False)
          Jrot_mat_nospatial = torch.matmul(U, Vh)
        elif self.polar_decomposition_mode == PolarDecompositionMode.NEWTON:
          Jrot_mat_nospatial = newton_iterate[self.newton_iteration_scale_factor](J_mat_nospatial, self.num_iterations)
        elif self.polar_decomposition_mode == PolarDecompositionMode.HALLEY:
          self.update_id_stack(J_mat_nospatial.shape[0], torch.device('cpu'))
          Jrot_mat_nospatial = halley_iterate(J_mat_nospatial.cpu(), self.num_iterations, self.id_stack).to(self.device)
        elif self.polar_decomposition_mode == PolarDecompositionMode.HALLEY_DYNAMIC_WEIGHTS:
          self.update_id_stack(J_mat_nospatial.shape[0], torch.device('cpu'))
          Jrot_mat_nospatial = halley_iterate_QDWH(J_mat_nospatial.cpu(), self.num_iterations, self.id_stack).to(self.device)
      elif self.tensor_transform_type == TensorTransformType.NONE:
        Jrot_mat_nospatial = J_mat_nospatial
      else:
        raise Exception(f"tensor_transform_type {self.tensor_transform_type} is not supported.")

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

  def update_id_stack(self, batch_size, device = None):
    """Update cached id_stack based on batch size"""
    if device is None:
      device = self.device
    if batch_size != self.last_id_stack_batch_size:
       self.id_stack = torch.repeat_interleave(torch.eye(3).unsqueeze(0), batch_size, dim=0).to(device)
    self.last_id_stack_batch_size = batch_size

class MseLossDTI(nn.Module):
  """Compare two DTIs and return the spatially averaged squared distance between diffusion tensors,
  where two diffusion tensors D1 and D2 are compared using the Frobenius norm, which is equivalent
  to tr((D1-D2)^2) because they are symmetric matrices.
  """
  def __init__(self, device='cpu') -> None:
    super().__init__()

    self.symmetrization_multiplier = torch.tensor([1,2,1,2,2,1]).view((1,-1,1,1,1)).to(device)

  def forward(self, b1: torch.Tensor, b2: torch.Tensor, weighting: torch.Tensor = None) -> torch.Tensor:
    """Input two batches of DTIs for comparison, each shaped (B,6,H,W,D).
    The channel dimension is interpreted to be the lower triangular entries of
    the diffusion tensors, in the ordering used by dipy: (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz).
    An optional weighting can be provided, shape (B,1,H,W,D), to weight the squared errors before taking the mean."""
    if weighting is not None:
      return (weighting * (((b1-b2)**2) * self.symmetrization_multiplier).sum(dim=1)).mean()
    return (((b1-b2)**2) * self.symmetrization_multiplier).sum(dim=1).mean()


def eig_dti(dti):
    """Compute the eignsystem from the DTI given in lower triangular form (B,6,H,W,D).
    Returns eigvals, eigvecs

    eigvals, shape (B,3,H,W,D), is the eigenvalues in ascending order,
    while eigvecs, shape (B,3,3,H,W,D), is the corresponding eigenvectors.
    e.g. eigvals[b,2,x,y,z] is the principal eigenvalue at (x,y,z)
    and eigvecs[b,2,:,x,y,z] is the associated eigenvector
    """

    b,c,h,w,d = dti.shape

    # move the spatial dimensions into the batch dimension,
    # and switch from lower traingular form to 3x3 symmetric matrices
    dti_spatially_batched = dti.permute((0,2,3,4,1)).reshape(-1,6)[:,[[0,1,3],[1,2,4],[3,4,5]]]

    # compute eigensystem
    eigvals, eigvecs = torch.linalg.eigh(dti_spatially_batched)

    # move spatial dimensions back out
    eigvals = eigvals.reshape(b,h,w,d,3).permute((0,4,1,2,3))
    eigvecs = eigvecs.reshape(b,h,w,d,3,3).permute((0,4,5,1,2,3))

    return eigvals, eigvecs


def fa_from_eigenvals(eigvals):
  """Compute the fractional anisotropy (FA) from the eigenvalues of a DTI image.
  eigvals image should have shape (B,3,H,W,D)
  Returns FA image of shape (B,1,H,W,D)
  """

  lambdahat = eigvals.sum(dim=1, keepdim=True)/3
  denom_sq = (eigvals**2).sum(dim=1, keepdim=True)
  numer_sq = ((eigvals - lambdahat)**2).sum(dim=1, keepdim=True)
  fa = (3/2)**(1/2) * (numer_sq / (denom_sq + 1e-20)).sqrt()
  return fa

def aoe_dti(dti1, dti2, fa1=None):
  """Given two DTIs and he FA image of the first one, return the AOE similarity measure of the two DTIs.
  AOE stands for "average overlap of eigenvectors"
  Sometimes it is also called OVL.
  See https://pubmed.ncbi.nlm.nih.gov/10893520/

  By construction, 0 < AOE <= 1, where 0 indicates no overlap and 1 indicates complete overlap.

  Args:
    dti1: DTI image in lower triangular form, shape (B,6,H,W,D)
    dti2: DTI image in lower triangular form, shape (B,6,H,W,D)
    fa1: the fa values associated to dti1, shape (B,1,H,W,D); will be computed if not provided

  Returns AOE values, shape (B,)
  """

  eigvals1, eigvecs1 = eig_dti(dti1)
  eigvals2, eigvecs2 = eig_dti(dti2)

  if fa1 is None:
    fa1 = fa_from_eigenvals(eigvals1)

  wm_mask = (fa1>=0.3)[:,0] # shape (B,H,W,D)
  numerator = ( eigvals1 * eigvals2 * ((eigvecs1 * eigvecs2).sum(dim=2)**2) ).sum(dim=1)
  denominator = (eigvals1 * eigvals2).sum(dim=1)
  frac = (numerator/denominator) # shape (B,H,W,D)

  out = []
  for b,img in enumerate(frac): # loop over batch; img has shape (H,W,D)
      out.append(img[wm_mask[b]].mean())
  aoe = torch.stack(out) # shape (B,)
  return aoe