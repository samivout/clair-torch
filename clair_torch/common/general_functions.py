"""
Module for general functions that are used across the project and that might come handy for users. Likely subject to
refactoring before 1.0.0 release.
"""
import math
from typing import Optional, Sequence, Any, Iterator, Type
from itertools import repeat
from collections.abc import Iterable
import yaml
import argparse

import torch
import numpy as np
from torchvision.transforms import GaussianBlur

from clair_torch.validation.type_checks import validate_all, is_broadcastable


def check_equal_lengths(*sequences: Optional[Sequence[Any]],
                        names: Optional[Sequence[str]] = None,
                        allow_none: bool = True) -> tuple[bool, dict[str, Optional[int]]]:
    """
    Checks whether all non-None sequences have the same length.

    Parameters:
        sequences: The sequences to check.
        names: Optional names for the sequences (used in result dictionary).
        allow_none: Whether None values are allowed. If False, None will cause a failure.

    Returns:
        A tuple (is_valid, lengths_dict) where:
            - is_valid is True if all non-None sequences have equal length (and no Nones if allow_none is False).
            - lengths_dict maps sequence names to their lengths (or None if the sequence was None).
    """
    if names is None:
        names = [f"seq{i}" for i in range(len(sequences))]
    if len(names) != len(sequences):
        raise ValueError("Length of 'names' must match number of sequences")

    lengths = {}
    for name, seq in zip(names, sequences):
        if seq is None:
            if not allow_none:
                return False, {name: None for name in names}
            lengths[name] = None
        else:
            lengths[name] = len(seq)

    non_none_lengths = {k: v for k, v in lengths.items() if v is not None}
    unique_lengths = set(non_none_lengths.values())

    return (len(unique_lengths) <= 1), lengths


def normalize_container(value: Any, target_type: Type = list, *, convert_if_iterable: bool = False,
                        exclude_types: tuple[type, ...] = (str, bytes, np.ndarray, torch.Tensor),
                        none_if_all_none: bool = True) -> Iterable | None:
    """
    Normalize a value into a container of the given type.

    Args:
        value: The input value to normalize.
        target_type: The container type to wrap the value in. Defaults to list.
        convert_if_iterable: whether to convert even if value is already iterable, but not of target_type.
        exclude_types: Types to exclude from being treated as iterable. Defaults to (str, bytes, np.ndarray, torch.Tensor)
        none_if_all_none: whether to return None if value is None or container contains only None(s).

    Returns:
        A container of type `target_type`, or None.
    """
    if value is None:
        return None if none_if_all_none else target_type()

    if isinstance(value, target_type):
        result = value
    elif isinstance(value, Iterable) and not isinstance(value, exclude_types):
        result = target_type(value) if convert_if_iterable else value
    else:
        result = target_type([value])

    if none_if_all_none and all(v is None for v in result):
        return None

    return result


def zip_with_none(*sequences: Optional[Sequence[Any]],
                  length: Optional[int] = None) -> Iterator[tuple[Any, ...]]:
    """
    Returns an iterator over items in the sequences. For None sequences, yields None.

    Parameters:
        sequences: The sequences to zip.
        length: Optional length to enforce. If not given, inferred from the first non-None sequence.

    Yields:
        Tuples with one item from each sequence. If a sequence is None, yields None in that position.
    """
    # Determine length from the first non-None sequence
    if length is None:
        for seq in sequences:
            if seq is not None:
                length = len(seq)
                break
        else:
            raise ValueError("Cannot determine length from all-None sequences and no length was provided.")

    # Build iterators, repeating None for any None sequence
    iterators = [(iter(seq) if seq is not None else repeat(None)) for seq in sequences]

    for _ in range(length):
        yield tuple(next(it) for it in iterators)


def weighted_mean_and_std(
        values: torch.Tensor, weights: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
        dim=None, keepdim=False, eps: float = 1e-8, compute_std: Optional[bool] = True) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the weighted mean and variance of `values`, with optional weights and boolean mask.
    Args:
        values (Tensor): Input tensor.
        weights (Tensor or None): Optional weights, broadcastable to `values`.
        mask (Tensor or None): Optional boolean mask, where True = valid value.
        dim (int or tuple of ints): Axis or axes to reduce over.
        keepdim (bool): Keep reduced dimensions.
        eps (float): Small value to avoid division by zero.
        compute_std: whether to compute std or not.

    Returns:
        (mean, variance): Tuple of tensors, each of the reduced shape.
    """
    std = None

    if mask is not None:

        mask = mask.to(dtype=values.dtype)
        values = values * mask
        weights = weights * mask if weights is not None else mask

    if weights is None:

        mean = values.mean(dim=dim, keepdim=True)

        if compute_std:
            sq_diff = (values - mean) ** 2
            std = torch.sqrt(sq_diff.mean(dim=dim, keepdim=True))

    else:

        weighted_sum = (values * weights).sum(dim=dim, keepdim=True)
        total_weight = weights.sum(dim=dim, keepdim=True).clamp(min=eps)

        # Handle edge case: total weight is zero (e.g. all weights are 0 or all masked)
        zero_weight_mask = (total_weight == 0)
        safe_weight = total_weight.clamp(min=eps)

        mean = weighted_sum / safe_weight

        if compute_std:
            sq_diff = (values - mean) ** 2
            weighted_sq_diff = (sq_diff * weights).sum(dim=dim, keepdim=True)
            std = torch.sqrt(weighted_sq_diff / safe_weight)

        # Set mean and std to 0 where total weight is zero
        mean = torch.where(zero_weight_mask, torch.zeros_like(mean), mean)
        if compute_std:
            std = torch.where(zero_weight_mask, torch.zeros_like(std), std)

    if not keepdim:
        mean = mean.squeeze(dim) if dim is not None else mean.squeeze()
        if compute_std:
            std = std.squeeze(dim) if dim is not None else std.squeeze()

    return mean, std


def flat_field_mean(flat_field: torch.Tensor, mid_area_side_fraction: float) -> torch.Tensor:
    """
    Computes the spatial mean over a centered square ROI for each image and channel.

    Args:
        flat_field (Tensor): Input tensor of shape (N, C, H, W)
        mid_area_side_fraction (float): Fraction of spatial dims to use for the ROI. Must lie in range [0.0, 1.0].

    Returns:
        Tensor: Mean over the ROI, shape (...)
    """
    if not isinstance(flat_field, torch.Tensor):
        raise TypeError(f"Expected flat_field as torch.Tensor, got {type(flat_field)}")
    if not isinstance(mid_area_side_fraction, float):
        raise TypeError(f"Expected mid_area_side_fraction as float, got {type(mid_area_side_fraction)}")
    if mid_area_side_fraction > 1.0 or mid_area_side_fraction < 0.0:
        raise ValueError(f"mid_area_side_fraction should be between 0.0 and 1.0")

    N, C, H, W = flat_field.shape

    ROI_dx = math.floor(W * mid_area_side_fraction)
    ROI_dy = math.floor(H * mid_area_side_fraction)

    ROI_start_index = (math.floor(1 / mid_area_side_fraction) - 1) / 2

    x_start = math.floor(ROI_start_index * ROI_dx)
    x_end = math.floor((ROI_start_index + 1) * ROI_dx)
    y_start = math.floor(ROI_start_index * ROI_dy)
    y_end = math.floor((ROI_start_index + 1) * ROI_dy)

    cropped = flat_field[:, :, y_start:y_end, x_start:x_end]  # shape: (N, C, ROI_dy, ROI_dx)

    return cropped.mean(dim=(-1, -2), keepdim=True)  # shape: (N, C, 1, 1)


def flatfield_correction(images: torch.Tensor, flatfield: torch.Tensor, flatfield_mean_val: torch.Tensor,
                         epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes a flat-field corrected version of input image by utilizing the given flat-field image and a given spatial
    mean. Ideally expects both images and flatfield in shape (N, C, H, W) but others are allowed. Match argument shapes
    based on requirements. For example with images (N, C, H, W) use flatfield (1, C, H, W) to apply same flatfield
    across the batch dimension, (N, 1, H, W) to apply unique flatfields across batch while disregarding channel specific
    features. Similarly, use flatfield_mean_val (1, C, 1, 1) to apply channel-specific scaling uniformly across the batch.

    Args:
        images: Image tensor of shape (N, C, H, W).
        flatfield: Flat field calibration image, same shape as `images` or broadcastable to images.
        flatfield_mean_val: Values used to scale the image. Match shape based on the given images and flatfield.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Tensor: Corrected image tensor, same shape as input.
    """
    validate_all((images, flatfield, flatfield_mean_val), torch.Tensor, raise_error=True, allow_none_iterable=False,
                 name="Tensor")
    validate_all([epsilon], float, raise_error=True, allow_none_iterable=False, name="Epsilon")

    is_broadcastable(images.shape, flatfield.shape, raise_error=True)
    is_broadcastable(images.shape, flatfield_mean_val.shape, raise_error=True)

    corrected_images = (images / (flatfield + epsilon)) * flatfield_mean_val

    return corrected_images


def get_valid_exposure_pairs(increasing_exposure_values: torch.Tensor, exposure_ratio_threshold: Optional[float] = None,
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate valid (i, j) index pairs for exposure comparison, based on a minimum ratio threshold.

    Args:
        increasing_exposure_values (Tensor): Shape (N,) exposure values in an increasing order.
        exposure_ratio_threshold (float, optional): Minimum exposure ratio to accept a pair.

    Returns:
        Tuple of:
            - i_idx: (P,) indices of first images in valid pairs (i < j)
            - j_idx: (P,) indices of second images
            - ratio_pairs: (P,) exposure ratios exposure[i] / exposure[j]
    """
    if not isinstance(increasing_exposure_values, torch.Tensor):
        raise TypeError(f"Expected increasing_exposure_values as torch.Tensor, got {type(increasing_exposure_values)}")
    if exposure_ratio_threshold is not None and not isinstance(exposure_ratio_threshold, float):
        raise TypeError(f"Expected exposure_ratio_threshold as float, got {type(exposure_ratio_threshold)}")

    N = increasing_exposure_values.shape[0]
    device = increasing_exposure_values.device
    ratios = increasing_exposure_values.view(N, 1) / increasing_exposure_values.view(1, N)  # (N, N)
    i_idx, j_idx = torch.triu_indices(N, N, offset=1)
    i_idx = i_idx.to(device=device)
    j_idx = j_idx.to(device=device)
    ratio_pairs = ratios[i_idx, j_idx]  # (P,)

    if exposure_ratio_threshold is not None:
        mask = ratio_pairs >= exposure_ratio_threshold
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        ratio_pairs = ratio_pairs[mask]

    return i_idx, j_idx, ratio_pairs


def get_pairwise_valid_pixel_mask(image_value_stack: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor,
                                  image_std_stack: Optional[torch.Tensor] = None,
                                  val_lower: float = 0.0, val_upper: float = 1.0,
                                  std_lower: float = 0.0, std_upper: float = 5e-4) -> torch.Tensor:
    """
    For a batch of images, for all pairs given by indices i_idx and j_idx, compute a pairwise boolean mask by marking
    invalid pixels as False, if they lie outside the valid range defined by lower and upper, in either one of the images
    in a given pair.
    Args:
        image_value_stack: batch of value images, shape (N, C, H, W).
        i_idx: the first set of indices to create pairs off of, shape (P,).
        j_idx: the second set of indices to create pairs off of, shape (P,).
        image_std_stack: batch of uncertainty images associated with the images in image_value_stack, shape (N, C, H, W).
        val_lower: lower threshold for marking pixels as invalid in value image.
        val_upper: upper threshold for marking pixels as invalid in value image.
        std_lower: lower threshold for marking pixels as invalid in std image.
        std_upper: upper threshold for marking pixels as invalid in std image.

    Returns:
        A boolean tensor that marks invalid pixel positions in a pair with False, shape (P, C, H, W).
    """
    validate_all((image_value_stack, i_idx, j_idx), torch.Tensor, allow_none_iterable=False, raise_error=True)
    validate_all((val_lower, val_upper, std_lower, std_upper), float, allow_none_iterable=False, raise_error=True)
    if image_std_stack is not None and not isinstance(image_std_stack, torch.Tensor):
        raise TypeError(f"Expected image_std_stack as torch.Tensor, got {type(image_std_stack)}")
    if val_lower > val_upper or std_lower > std_upper:
        raise ValueError("Lower threshold cannot be a larger value than upper threshold.")

    # Gather image pairs and reshape to (P, C, H, W)
    val_i, val_j = image_value_stack[i_idx], image_value_stack[j_idx]

    valid_mask = (val_i >= val_lower) & (val_i <= val_upper) & (val_j >= val_lower) & (val_j <= val_upper)

    if image_std_stack is not None:
        std_i, std_j = image_std_stack[i_idx], image_std_stack[j_idx]
        valid_std_mask = (std_i >= std_lower) & (std_i <= std_upper) & (std_j >= std_lower) & (std_j <= std_upper)
        valid_mask = valid_mask & valid_std_mask

    return valid_mask


def cv_to_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Function for transforming a tensor with OpenCV channel and dimension ordering into PyTorch channel and dimension
    ordering. Expects a tensor with two or three dimensions.
    Args:
        x: tensor to transform.

    Returns:
        torch.Tensor
    """
    if x.ndim == 2:
        # Grayscale image: (H, W) -> (1, H, W)
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[2] == 3:
        # Color image: BGR to RGB, then permute to (C, H, W)
        x = x[:, :, [2, 1, 0]]  # BGR to RGB
        x = x.permute(2, 0, 1)
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    return x


def torch_to_cv(x: torch.Tensor) -> torch.Tensor:

    if x.ndim == 2:
        # Already single-channel grayscale: no change
        return x

    elif x.ndim == 3 and x.shape[0] == 3:
        # Color image: (C, H, W) -> (H, W, C), then RGB to BGR
        cv_format = x.permute(1, 2, 0)  # to HWC
        cv_format = cv_format[:, :, [2, 1, 0]]  # RGB to BGR
        return cv_format

    elif x.ndim == 3 and x.shape[0] == 1:
        # Single-channel: (1, H, W) -> (H, W)
        cv_format = x.squeeze(0)
        return cv_format

    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")


def normalize_tensor(x: torch.Tensor, max_val: Optional[float] = None, min_val: Optional[float] = None,
                     target_range: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """
    Normalize a tensor by a given min and max value. If not provided, uses the min and max of the tensor.

    Args:
        x: Input tensor.
        max_val: Optional maximum value for normalization.
        min_val: Optional minimum value for normalization.
        target_range: Tuple specifying the (min, max) target range.

    Returns:
        The normalized tensor.
    """
    max_val = x.max() if max_val is None else max_val
    min_val = x.min() if min_val is None else min_val

    denominator = max_val - min_val
    if denominator == 0:
        raise ValueError("Normalization range is zero (min == max); cannot normalize.")

    # Normalize to [0, 1]
    x_normalized = (x - min_val) / denominator

    # Scale to [target_min, target_max]
    min_target, max_target = target_range
    target_span = max_target - min_target
    x_scaled = x_normalized * target_span + min_target

    return x_scaled


def clamp_along_dims(x: torch.Tensor, dim: int | tuple[int, ...],
                     min_max_pairs: tuple[float, float] | list[tuple[float, float]]) -> torch.Tensor:
    """
    Clamp a tensor along specified dimension(s).

    Args:
        x: Input tensor.
        dim: int or tuple of ints, dimensions along which to apply min/max.
        min_max_pairs: Single tuple (min, max) applied to all slices. List of tuples; length must match the number of
            slices along dims.

    Returns:
        Clamped tensor of same shape as x.
    """
    if isinstance(dim, int):
        dim = (dim,)

    validate_all([x], torch.Tensor, raise_error=True, allow_none_elements=False, allow_none_iterable=False)
    validate_all(dim, int, raise_error=True, allow_none_elements=False, allow_none_iterable=False)

    dim = tuple(d % x.ndim for d in dim)  # handle negative dims

    # Determine slice shape along specified dims
    slice_shape = tuple(x.shape[d] for d in dim)
    num_slices = torch.tensor(slice_shape).prod().item()

    # Handle single min-max pair
    if isinstance(min_max_pairs, tuple):
        return torch.clamp(x, min=min_max_pairs[0], max=min_max_pairs[1])

    # Otherwise we expect a list of min-max pairs
    if len(min_max_pairs) != num_slices:
        raise ValueError(f"Expected 1 or {num_slices} min/max pairs, got {len(min_max_pairs)}")

    # Convert min/max pairs to tensors of shape slice_shape
    mins = torch.tensor([pair[0] for pair in min_max_pairs], dtype=x.dtype).reshape(slice_shape)
    maxs = torch.tensor([pair[1] for pair in min_max_pairs], dtype=x.dtype).reshape(slice_shape)

    # Create broadcasting shape
    broadcast_shape = [1] * x.ndim
    for i, d in enumerate(dim):
        broadcast_shape[d] = slice_shape[i]

    mins = mins.reshape(broadcast_shape)
    maxs = maxs.reshape(broadcast_shape)

    # Clamp using broadcasting
    return torch.clamp(x, min=mins, max=maxs)


def conditional_gaussian_blur(image: torch.Tensor, mask_map: torch.Tensor, threshold: float, kernel_size: int) -> torch.Tensor:
    """
    Apply a gaussian blur on input image positions at which the given map has value larger than the given threshold.
    Main purpose is to filter hot pixels from an input image according to a dark calibration image.
    Args:
        image: input image to filter.
        mask_map: map upon whose values the filtering is based on.
        threshold: the threshold value to apply filtering on any given position.
        kernel_size: size of the gaussian blur kernel.

    Returns:
        The filtered image.
    """
    *leading, C, H, W = image.shape
    image_flat = image.reshape(-1, C, H, W)
    N = image_flat.size(0)

    blur_transform = GaussianBlur(kernel_size=kernel_size, sigma=1.0)
    blurred = blur_transform(image)

    # boolean mask with stride-0 in the batch dimension: shape (1, C, H, W)
    mask = (mask_map > threshold).unsqueeze(0)

    out = torch.where(mask, blurred, image_flat)

    return out.reshape(*leading, C, H, W)


def cli_parse_args_from_config() -> dict:
    """
    CLI utility function to read parameter values from a .yaml config file into a dictionary.
    Returns:
        dictionary of the parsed keys and values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    return config
