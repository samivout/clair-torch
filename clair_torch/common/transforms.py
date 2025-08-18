"""
Module for transforms to be used with PyTorch tensors. Main use is for chaining required operations on in __getitem__
of ImageDataset class.
"""
from typing import Protocol, Optional, Tuple, runtime_checkable

import torch

from clair_torch.common.general_functions import (conditional_gaussian_blur, cv_to_torch, torch_to_cv, normalize_tensor,
                                                  clamp_along_dims)
from clair_torch.validation.type_checks import validate_all


@runtime_checkable
class Transform(Protocol):
    """
    Class stub for typing purposes. Used to indicate that a transform such as the ones in this module is required.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class CvToTorch:
    """
    Transform for modifying a tensor from OpenCV dimensions and channel ordering to a PyTorch dimensionality and
    ordering.
    """

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return cv_to_torch(x)


class TorchToCv:
    """
    Transform for converting a tensor from PyTorch (C, H, W) format with RGB channels to OpenCV format (H, W, C) with
    BGR channels.
    """

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return torch_to_cv(x)


class Normalize:
    """
    Transform for normalizing a tensor by a minimum and maximum value.
    If no values are given, dynamically use min and max of the given tensor.
    """
    def __init__(
        self,
        max_val: Optional[float] = None,
        min_val: Optional[float] = None,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ):
        validate_all([max_val, min_val], (float, int), raise_error=True, allow_none_elements=True)
        validate_all(target_range, float, raise_error=True, allow_none_elements=False)

        self.max_val = max_val
        self.min_val = min_val
        self.target_range = target_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(x, max_val=self.max_val, min_val=self.min_val, target_range=self.target_range)


class ClampAlongDims:
    """
    Transform for clamping the tensor values between a min and max value, along the given dimension(s).
    """
    def __init__(self, dim: int | tuple[int, ...], min_max_pairs: tuple[float, float] | list[tuple[float, float]]):

        self.dim = dim
        self.min_max_pairs = min_max_pairs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return clamp_along_dims(x, self.dim, self.min_max_pairs)


class CastTo:
    """
    Transform for casting the tensor to the given datatype and device. If data_type or device are not give, then
    maintain them as they are.
    """
    def __init__(self, data_type: Optional[torch.dtype] = None, device: Optional[torch.device] = None):

        validate_all([data_type], torch.dtype, raise_error=True, allow_none_elements=True)
        validate_all([device], torch.device, raise_error=True, allow_none_elements=True)

        self.data_type = data_type
        self.device = device

    def __call__(self, x: torch.Tensor):

        data_type = self.data_type if self.data_type is not None else x.dtype
        device = self.device if self.device is not None else x.device

        return x.to(dtype=data_type, device=device)


class StridedDownScale:
    """
    Transform for applying a spatial downscale on the input tensor.
    """
    def __init__(self, step_size: int):

        validate_all([step_size], int, raise_error=True, allow_none_elements=False)
        if step_size < 0:
            raise ValueError(f"step_size must be non-negative.")

        self.step_size = step_size

    def __call__(self, x: torch.Tensor):

        strided = x[..., ::self.step_size, ::self.step_size]

        return strided


class BadPixelCorrection:

    def __init__(self, bad_pixel_map: torch.Tensor, threshold: float, kernel_size: int):

        validate_all([bad_pixel_map], torch.Tensor, raise_error=True, allow_none_elements=False)
        validate_all([threshold], float, raise_error=True, allow_none_elements=False)
        validate_all([kernel_size], int, raise_error=True, allow_none_elements=False)
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be larger than 0, got {kernel_size}")

        self.bad_pixel_map = bad_pixel_map
        self.threshold = threshold
        self.kernel_size = kernel_size

    def __call__(self, x: torch.Tensor):

        filtered = conditional_gaussian_blur(x, self.bad_pixel_map, self.threshold, self.kernel_size)

        return filtered

    def clear_memory(self):

        self.bad_pixel_map = None


if __name__ == "__main__":
    pass
