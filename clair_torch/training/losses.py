"""
Module for loss computations. Contains loss functions for image linearity related aspects, as well as penalty functions
for the ICRF curve that the training loop solves.
"""
from typing import Optional

import torch

from clair_torch.common.general_functions import weighted_mean_and_std
from clair_torch.validation.type_checks import validate_all, validate_dimensions, validate_multiple_dimensions


def pixelwise_linearity_loss(
    image_value_stack: torch.Tensor,     # (N, C, H, W)
    i_idx: torch.Tensor,                 # (P,)
    j_idx: torch.Tensor,                 # (P,)
    ratio_pairs: torch.Tensor,           # (P,)
    image_std_stack: Optional[torch.Tensor] = None,
    use_relative: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute a differentiable linearity loss for a stack of images taken at different exposures.
    Args:
        image_value_stack: image stack of shape (N, C, H, W), values in [0, 1]
        exposure_values: 1D tensor of shape (N,), exposure values
        image_std_stack: optional standard deviation estimates, same shape as image_value_stack
        lower: pixel intensity threshold (inclusive)
        upper: pixel intensity threshold (inclusive)
        use_relative: compute relative difference instead of absolute
        exposure_ratio_threshold: threshold value, under which exposure ratios are rejected.

    Returns:
        A scalar loss (mean deviation from expected linearity across valid pixels and channel pairs)
    """

    # Gather image pairs and reshape to (P, C, H, W)
    I_i = image_value_stack[i_idx]  # (P, C, H, W)
    I_j = image_value_stack[j_idx]  # (P, C, H, W)

    # Apply ratio: expect I_i ≈ I_j * ratio
    expected = I_j * ratio_pairs.view(-1, 1, 1, 1)

    diff = I_i - expected
    if use_relative:
        # Avoid division by zero — small epsilon
        expected_safe = expected + 1e-6
        diff = diff / expected_safe

    abs_diff = diff.abs()

    if image_std_stack is not None:
        std_i = image_std_stack[i_idx]
        std_j = image_std_stack[j_idx]
        if use_relative:
            eps = 1e-6
            # expected_safe = expected.clamp(min=eps)
            I_j_safe = I_j.clamp(min=eps)

            term1 = (std_i / expected_safe) ** 2
            term2 = ((I_i * std_j) / (expected_safe * I_j_safe)) ** 2

            std = torch.sqrt(term1 + term2 + eps)
        else:
            std = torch.sqrt(std_i ** 2 + (ratio_pairs.view(-1, 1, 1, 1) * std_j) ** 2)
    else:
        std = None

    return abs_diff, std


def compute_spatial_linearity_loss(pixelwise_losses: torch.Tensor, pixelwise_errors: Optional[torch.Tensor] = None,
                                   external_weights: Optional[torch.Tensor] = None,
                                   valid_mask: Optional[torch.Tensor] = None, use_uncertainty_weighting: bool = True) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Computes the spatial mean, standard deviation and uncertainty based on the given pixelwise losses and
    possible pixelwise uncertainties. External weights and a valid mask can be used to modify the computation
    as needed.
    Args:
        pixelwise_losses: tensor containing the pixelwise linearity losses.
        pixelwise_errors: tensor containing the uncertainty of the pixelwise linearity losses.
        external_weights: tensor containing external weight values for mean computation.
        valid_mask: boolean tensor representing pixel positions that are valid for the computation.
        use_uncertainty_weighting: whether to utilize inverse uncertainty based weights.

    Returns:
        Tuple representing (the spatial mean of linearity loss, the spatial standard deviation of linearity loss,
        spatial uncertainty of the linearity loss).
    """

    # Stage the use of weighting. Uncertainty weighting utilizes the computed uncertainties as inverse weights
    # prioritize values with smaller uncertainties. Possible external weights are added to the possible
    # uncertainty-based weights.
    if pixelwise_errors is not None or external_weights is not None:
        weights = torch.zeros_like(pixelwise_losses)
        if pixelwise_errors is not None and use_uncertainty_weighting:
            weights = weights + (1 / (pixelwise_errors + 1e-6))
        if external_weights is not None:
            weights = weights + external_weights
    else:
        weights = None

    spatial_linearity_loss, spatial_linearity_loss_std = weighted_mean_and_std(pixelwise_losses, weights=weights, mask=valid_mask, dim=(2, 3))
    if pixelwise_errors is not None:
        spatial_linearity_loss_error, _ = weighted_mean_and_std(pixelwise_errors, mask=valid_mask, dim=(2, 3))
    else:
        spatial_linearity_loss_error = None

    return spatial_linearity_loss, spatial_linearity_loss_std, spatial_linearity_loss_error


def compute_monotonicity_penalty(curve, squared=True, per_channel: bool = False) -> torch.Tensor:
    """
    Get a penalty term for a function (tensor) if it is not monotonically increasing.
    # TODO: flip dimensions so that channels are the leftmost dimension.
    Args:
        curve: the function to determine a penalty term for.
        squared: whether to square the penalty terms.
        per_channel: whether to return a per-channel loss, or sum the channel losses into a scalar tensor.
    Returns:
        Penalty terms as scalar tensor, or one element per channel.
    """
    df = curve[1:, :] - curve[:-1, :]  # Shape: (N-1, C)
    mask = (df <= 0).float()  # 1.0 where non-strictly increasing
    if squared:
        relu_neg = mask * df.pow(2)  # penalize squared difference
    else:
        relu_neg = mask * (-df)  # penalize linearly

    penalty = relu_neg.sum(dim=0)  # Shape: (C,)

    if per_channel:
        return penalty
    return torch.sum(penalty)


def compute_smoothness_penalty(curve: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    """
    Get a penalty term for a function (tensor) if it is not sufficiently smooth.
    # TODO: flip dimensions so that channels are the leftmost dimension.
    Args:
        curve:
        per_channel: whether to return a per-channel loss, or sum the channel losses into a scalar tensor.
    Returns:
        Penalty terms as scalar tensor, or one element per channel.
    """
    second_diff = curve[:-2, :] - 2 * curve[1:-1, :] + curve[2:, :]
    penalty = second_diff.pow(2).sum(dim=0)  # Shape: [3]
    if per_channel:
        return penalty
    return torch.sum(penalty)


def compute_range_penalty(curve: torch.Tensor, epsilon: float = 1e-6, per_channel: bool = False) -> torch.Tensor:
    """
    Get a penalty term for a function (tensor) if it breaks [0, 1] value range.
    # TODO: flip dimensions so that channels are the leftmost dimension.
    Args:
        curve: the function to determine a penalty term for.
        epsilon: small epsilon to avoid vanishing gradients.
        per_channel: whether to return a per-channel loss, or sum the channel losses into a scalar tensor.

    Returns:
        Penalty terms as scalar tensor, or one element per channel.
    """
    validate_all([curve], torch.Tensor, allow_none_elements=False, allow_none_iterable=False, raise_error=True)
    validate_all([epsilon], float, allow_none_elements=False, allow_none_iterable=False, raise_error=True)

    lower = torch.relu(-curve)      # Penalize values under 0
    upper = torch.relu(curve - 1)   # Penalize values over 1
    penalty = (lower + upper).sum(dim=0)  # Sum over 256, shape: [3]

    if per_channel:
        return penalty
    return torch.sum(penalty)


def compute_endpoint_penalty(curve: torch.Tensor, per_channel: Optional[bool] = False) -> torch.Tensor:
    """
    Get a penalty term for a function (tensor), whose endpoints are not exactly 0 and 1. The penalty is defined as the
    sum of the squares of the deviations at start and end from 0 and 1 respectively.
    # TODO: flip dimensions so that channels are the leftmost dimension.
    Args:
        curve: curve: the function to determine a penalty term for. Shape (D, C) with D and C representing the number
            of datapoints in the function and C representing the number of channels.
        per_channel: whether to return a per-channel loss, or sum the channel losses into a scalar tensor.
    Returns:
        Penalty terms as scalar tensor, or one element per channel.
    """
    if curve.ndim == 1:
        curve = curve.unsqueeze(1)
    validate_all([curve], torch.Tensor, allow_none_iterable=False, allow_none_elements=False, raise_error=True)
    validate_dimensions(curve, (1, 2), raise_error=True)

    penalty = (curve[0, :] - 0) ** 2 + (curve[-1, :] - 1) ** 2  # - 0 to emphasize the definition of the loss.
    if per_channel:
        return penalty
    return torch.sum(penalty)


def gaussian_value_weights(image: torch.Tensor, scale: Optional[float] = 30.0) -> torch.Tensor:
    """
    Compute Gaussian weights for an image based on intensity proximity to 0.5.

    Args:
        image: torch.Tensor of shape (N, C, H, W), values in [0, 1]
        scale: float, controls sharpness of the Gaussian (default 30)

    Returns:
        weight: torch.Tensor of shape (N, C, H, W)
    """
    validate_all([image], torch.Tensor, allow_none_iterable=False, allow_none_elements=False, raise_error=True)
    validate_all([scale], float, allow_none_iterable=False, allow_none_elements=False, raise_error=True)

    return torch.exp(-scale * (image - 0.5) ** 2)


def combined_gaussian_pair_weights(
    image_stack: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: Optional[float] = 10.0
) -> torch.Tensor:
    """
    Compute combined Gaussian weights for image pairs by summing weights from each image.

    Args:
        image_stack: stack of images, ndim >= 2. Usual shape (N, C, H, W)
        i_idx: Index tensor of shape (P,) for first image in each pair, ndim=1.
        j_idx: Index tensor of shape (P,) for second image in each pair, ndim=1.
        scale: Sharpness of Gaussian weight.

    Returns:
        combined_weights: Tensor of shape (P, C, H, W) for input of (N, C, H, W).
    """
    validate_all([image_stack, i_idx, j_idx], torch.Tensor, allow_none_iterable=False, allow_none_elements=False,
                 raise_error=True)
    validate_all([scale], float, allow_none_iterable=False, allow_none_elements=False, raise_error=True)
    validate_multiple_dimensions([i_idx, j_idx], [1, 1])

    image_i = image_stack[i_idx]  # (P, C, H, W)
    image_j = image_stack[j_idx]  # (P, C, H, W)

    weights_i = gaussian_value_weights(image_i, scale)
    weights_j = gaussian_value_weights(image_j, scale)

    combined_weights = weights_i + weights_j
    return combined_weights

