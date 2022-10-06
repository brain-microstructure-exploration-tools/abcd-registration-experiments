import numpy as np
import matplotlib.pyplot as plt
import monai
import torch


def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None, slices=None):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    if slices is None:
        # half-way slices
        x, y, z = np.array(image_array.shape)//2
    else:
        x, y, z = slices
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im in zip(axs, imgs):
        ax.axis('off')
        ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin='lower')

    plt.show()


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, downsampling=None, slices = None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    if slices is None:
        # half-way slices
        x, y, z = np.array(vector_field.shape[1:])//2
    else:
        x, y, z = slices

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    plt.show()


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H,W))
    grid_img[np.arange(0, H, grid_spacing),:]=1
    grid_img[:,np.arange(0, W, grid_spacing)]=1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0) # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin='lower', cmap='gist_gray')


def preview_3D_deformation(vector_field, grid_spacing, slices=None, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    if slices is None:
        x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    else:
        x, y, z = slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.show()


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return np.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det


checkerboard = lambda i1, i2, e1, e2, n1, n2 : (-1)**(int((i1/e1)*n1) + int((i2/e2)*n2))

def preview_checkerboard(
    img1, img2, normalize_by="volume", cmap=None, figsize=(12, 12),
    slices=None, numSquares=8):
    """
    Interleave two 3D images in a checkerboard, and display as three orthogonal slices.

    img1 and img2 are assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = max(img1.max().item(), img2.max().item())
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    assert((img1.shape==img2.shape))

    if slices is None:
        # half-way slices
        x, y, z = np.array(img1.shape)//2
    else:
        x, y, z = slices

    slices1 = (img1[x, :, :], img1[:, y, :], img1[:, :, z])
    slices2 = (img2[x, :, :], img2[:, y, :], img2[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im1, im2 in zip(axs, slices1, slices2):
        e1,e2 = im1.shape
        cb = np.array([[checkerboard(i1,i2,e1,e2,numSquares,numSquares) for i2 in range(e2)] for i1 in range(e1)])
        cb_mask = (cb==1)

        assert(im1.shape==im2.shape)
        assert(im1.shape==cb.shape)

        im3 = np.zeros_like(im1)
        im3[cb_mask] = im1[cb_mask]
        im3[~cb_mask] = im2[~cb_mask]

        ax.axis('off')
        ax.imshow(im3, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

    plt.show()



def mse_loss(b1, b2):
    """Return image similarity loss given two batches b1 and b2 of shape (batch_size, channels, H,W,D).
    It is scaled up a bit here."""
    return ((b1-b2)**2).mean()

def ncc_loss(b1, b2):
    """Return the negative NCC loss given two batches b1 and b2 of shape (batch_size, channels, H,W,D).
    It is averaged over batches and channels."""
    mu1 = b1.mean(dim=(2,3,4)) # means
    mu2 = b2.mean(dim=(2,3,4))
    alpha1 = (b1**2).mean(dim=(2,3,4)) # second moments
    alpha2 = (b2**2).mean(dim=(2,3,4))
    alpha12 = (b1*b2).mean(dim=(2,3,4)) # cross term
    numerator = alpha12 - mu1*mu2
    denominator = torch.sqrt((alpha1 - mu1**2) * (alpha2-mu2**2))
    ncc = numerator / denominator
    return -ncc.mean() # average over batches and channels

def compose_ddf(u,v, warp):
    """Compose two displacement fields, return the displacement that warps by v followed by u
    Use the given warper (e.g. a monai.networks.blocks.Warp)"""
    return u + warp(v,u)

# Compute discrete spatial derivatives
def diff_and_trim(array, axis, spatial_size):
    """Take the discrete difference along a spatial axis, which should be 2,3, or 4.
    Return a difference tensor with all spatial axes trimmed by 1."""
    return torch.diff(array, axis=axis)[:, :, :(spatial_size[0]-1), :(spatial_size[1]-1), :(spatial_size[2]-1)]

def size_of_spatial_derivative(u):
    """Return the squared Frobenius norm of the spatial derivative of the given displacement field.
    To clarify, this is about the derivative of the actual displacement field map, not the deformation
    that the displacement field map defines. The expected input shape is (batch,3,H,W,D).
    Output shape is (batch)."""
    dx = diff_and_trim(u, 2)
    dy = diff_and_trim(u, 3)
    dz = diff_and_trim(u, 4)
    return(dx**2 + dy**2 + dz**2).sum(axis=1).mean(axis=[1,2,3])

# Monai transforms typically apply to individual objects and not batches.
# This functor takes a transform and produces a version of it that operates on batches.
# Handles MetaTensor correctly
class BatchifiedTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, batch):
        return monai.data.utils.list_data_collate(
            [self.transform(x) for x in monai.transforms.Decollated()(batch)]
        )

def batchify(f):
    """Return a function that maps the given function f over batches.

    Args:
        f: a function that maps torch tensors to torch tensors, with no batch dimension expected.

    This handles MetaTensor correctly."""
    def batchified_f(t):
        return monai.data.utils.list_data_collate(
            [f(x) for x in monai.transforms.Decollated()(t)]
        )
    
    return batchified_f

