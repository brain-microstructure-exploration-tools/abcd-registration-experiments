import warnings
from copy import deepcopy
from enum import Enum
from typing import Any, Mapping, Hashable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import USE_COMPILED, DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, iter_patch, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.networks.utils import meshgrid_ij, normalize_transform
from monai.transforms.croppad.array import CenterSpatialCrop, ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform, MapTransform
from monai.transforms.utils import (
    convert_pad_mode,
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
    scale_affine,
)
from monai.transforms.utils_pytorch_numpy_unification import allclose, linalg_inv, moveaxis, where
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
    pytorch_after,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import GridPatchSort, PytorchPadMode, TraceKeys, TransformBackends
from monai.utils.misc import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, get_equivalent_dtype, get_torch_dtype_from_string

from monai.transforms.spatial.array import Affine, RandRange, AffineGrid, Resample

from monai.data.utils import list_data_collate
from monai.data import decollate_batch


class AffineAugmentation:
    """Given two batches of 3D images, apply the same random affine transform to both, and then one different
    random affine transform to one of them. Useful augmentation for training affine registration models,
    where one may want to affinely transform a moving and target image pair together and then separately
    apply a transform to the moving image to displace it away from the target. The first transform helps the
    network learn to deal with different arrangements of the objects in the image space, and the second
    transform helps the network deal with different relative object configurations.

    If the second transform is omitted then this simply applies the same transform to both images."""
    def __init__(self, spatial_size, transformation_strength_both, transformation_strength_second = None):
        self.spatial_size = spatial_size
        self.do_both = self.make_transform_of_strength(transformation_strength_both)
        if transformation_strength_second is not None:
            self.do_second = self.make_transform_of_strength(transformation_strength_second)
        else:
            self.do_second = None

    def make_transform_of_strength(self, a):
        return RandAffineCustom(
            prob = 1.,
            mode = 'bilinear',
            padding_mode = 'zeros',
            spatial_size = self.spatial_size,
            cache_grid = True,
            rotate_range = (a*np.pi/2,)*3,
            shear_range = (0,)*6, # no shearing
            translate_range =  [a*s/16 for s in self.spatial_size],
            scale_range = (1.5**a,),
            scale_isotropically = True,
        )

    def __call__(self, first, second):
        """Input two batches of images, each of shape (b,c,h,w,d)."""
        out1 = []
        out2 = []
        for img1, img2 in zip(decollate_batch(first), decollate_batch(second)):
            out1.append(self.do_both(img1))
            img2_transformed = self.do_both(img2, randomize=False)
            if self.do_second is not None:
                img2_transformed = self.do_second(img2_transformed)
            out2.append(img2_transformed)
        return list_data_collate(out1), list_data_collate(out2)




# Copied and modified from monai.transforms.spatial.RandAffineGrid
# The only difference is how random scales are generated, with the additional of the parameter scale_isotropically
class RandAffineGridCustom(Randomizable, Transform):
    """
    Generate randomised affine grid.

    """

    backend = AffineGrid.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        scale_isotropically: bool = False,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxels to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. Logarithmically-uniformly generate scales.
                If scale_isotropically==True, then only one value (number or interval) needs to be given in scale_range
            TODO: document scale_isotropically
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        self.device = device
        self.affine: Optional[torch.Tensor] = torch.eye(4, dtype=torch.float64)

        self.scale_isotropically = scale_isotropically

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def _get_rand_param_scale(self, param_range):
      if self.scale_isotropically:
        f = param_range[0]
        if issequenceiterable(f):
            if len(f) != 2:
                raise ValueError("If giving range as [min,max], should only have two elements per dim.")
            s = np.exp(self.R.uniform(np.log(f[0]), np.log(f[1])))
        elif f is not None:
            lf = np.log(f)
            s = np.exp(self.R.uniform(-lf, lf))
        s = float(s)
        return [s,s,s]

      out_param = []
      for f in param_range:
          if issequenceiterable(f):
              if len(f) != 2:
                  raise ValueError("If giving range as [min,max], should only have two elements per dim.")
              out_param.append(np.exp(self.R.uniform(np.log(f[0]), np.log(f[1]))))
          elif f is not None:
              lf = np.log(f)
              out_param.append(np.exp(self.R.uniform(-lf, lf)))
      out_param = [float(s) for s in out_param]
      return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param_scale(self.scale_range)

    def __call__(
        self,
        spatial_size: Optional[Sequence[int]] = None,
        grid: Optional[NdarrayOrTensor] = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            randomize: boolean as to whether the grid parameters governing the grid should be randomized.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        if randomize:
            self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
        )
        _grid: torch.Tensor
        _grid, self.affine = affine_grid(spatial_size, grid)
        return _grid

    def get_transformation_matrix(self) -> Optional[torch.Tensor]:
        """Get the most recently applied transformation matrix"""
        return self.affine


# Copied and modified from monai.transforms.spatial.RandAffine
# The only difference is that it uses the above customized RandAffineGrid
class RandAffineCustom(RandomizableTransform, InvertibleTransform):
    """
    Random affine transform.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Affine.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[str, int] = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        scale_isotropically: bool = False,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``bilinear``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``reflection``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            TODO document change to scale_range and document scale_isotropically

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        RandomizableTransform.__init__(self, prob)

        self.rand_affine_grid = RandAffineGridCustom(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            scale_isotropically=scale_isotropically,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache()
        self.mode = mode
        self.padding_mode: str = padding_mode

    def _init_identity_cache(self):
        """
        Create cache of the identity grid if cache_grid=True and spatial_size is known.
        """
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'."
                )
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return create_grid(spatial_size=_sp_size, device=self.rand_affine_grid.device, backend="torch")

    def get_identity_grid(self, spatial_size: Sequence[int]):
        """
        Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(
            spatial_size, [2] * ndim
        ):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return (
            create_grid(spatial_size=spatial_size, device=self.rand_affine_grid.device, backend="torch")
            if self._cached_grid is None
            else self._cached_grid
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAffineCustom":
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[str, int, None] = None,
        padding_mode: Optional[str] = None,
        randomize: bool = True,
        grid=None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to execute `randomize()` function first, default to True.
            grid: precomputed grid to be used (mainly to accelerate `RandAffined`).

        """
        if randomize:
            self.randomize()
        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and device
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        do_resampling = self._do_transform or (sp_size != ensure_tuple(img.shape[1:]))
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if not do_resampling:
            out: torch.Tensor = convert_data_type(img, dtype=torch.float32, device=self.resampler.device)[0]
        else:
            if grid is None:
                grid = self.get_identity_grid(sp_size)
                if self._do_transform:
                    grid = self.rand_affine_grid(grid=grid, randomize=randomize)
            out = self.resampler(img=img, grid=grid, mode=_mode, padding_mode=_padding_mode)
        mat = self.rand_affine_grid.get_transformation_matrix()
        out = convert_to_tensor(out, track_meta=get_track_meta())
        if get_track_meta():
            self.push_transform(
                out,
                orig_size=img.shape[1:],
                extra_info={
                    "affine": mat,
                    "mode": _mode,
                    "padding_mode": _padding_mode,
                    "do_resampling": do_resampling,
                },
            )
            self.update_meta(out, mat, img.shape[1:], sp_size)
        return out

    def update_meta(self, img, mat, img_size, sp_size):
        affine = convert_data_type(img.affine, torch.Tensor)[0]
        img.affine = Affine.compute_w_affine(affine, mat, img_size, sp_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # if transform was not performed nothing to do.
        if not transform[TraceKeys.EXTRA_INFO]["do_resampling"]:
            return data
        orig_size = transform[TraceKeys.ORIG_SIZE]
        orig_size = fall_back_tuple(orig_size, data.shape[1:])
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(fwd_affine)
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]
        affine_grid = AffineGrid(affine=inv_affine)
        grid, _ = affine_grid(orig_size)

        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        self.update_meta(out, inv_affine, data.shape[1:], orig_size)
        return out  # type: ignore






class RandAffineCustomD(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAffine`.
    """

    backend = RandAffineCustom.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_isotropically: bool = False,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            TODO: document scale_isotropically
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_affine = RandAffineCustom(
            prob=1.0,  # because probability handled in this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            cache_grid=cache_grid,
            device=device,
            scale_isotropically=scale_isotropically
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAffineCustomD":
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        first_key: Union[Hashable, List] = self.first_key(d)
        if first_key == []:
            out: Dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        # all the keys share the same random Affine factor
        self.rand_affine.randomize()

        spatial_size = d[first_key].shape[1:]  # type: ignore
        sp_size = fall_back_tuple(self.rand_affine.spatial_size, spatial_size)
        # change image size or do random transform
        do_resampling = self._do_transform or (sp_size != ensure_tuple(spatial_size))
        # converting affine to tensor because the resampler currently only support torch backend
        grid = None
        if do_resampling:  # need to prepare grid
            grid = self.rand_affine.get_identity_grid(sp_size)
            if self._do_transform:  # add some random factors
                grid = self.rand_affine.rand_affine_grid(grid=grid)

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            # do the transform
            if do_resampling:
                d[key] = self.rand_affine(d[key], mode=mode, padding_mode=padding_mode, grid=grid)
            if get_track_meta():
                xform = self.pop_transform(d[key], check=False) if do_resampling else {}
                self.push_transform(d[key], extra_info={"do_resampling": do_resampling, "rand_affine_info": xform})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            tr = self.pop_transform(d[key])
            do_resampling = tr[TraceKeys.EXTRA_INFO]["do_resampling"]
            if do_resampling:
                d[key].applied_operations.append(tr[TraceKeys.EXTRA_INFO]["rand_affine_info"])  # type: ignore
                d[key] = self.rand_affine.inverse(d[key])

        return d