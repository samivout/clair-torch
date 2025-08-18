"""
The training subpackage provides a training loop function for the ICRF models. In addition to the training loop, it also
provides the various functions for loss computation used in the training loop.
"""

from .losses import gaussian_value_weights, combined_gaussian_pair_weights, pixelwise_linearity_loss, compute_spatial_linearity_loss, compute_range_penalty, compute_endpoint_penalty, compute_monotonicity_penalty, compute_smoothness_penalty
from .icrf_training import train_icrf

__all__ = [
    "gaussian_value_weights", "combined_gaussian_pair_weights", "pixelwise_linearity_loss", "compute_spatial_linearity_loss",
    "compute_smoothness_penalty", "compute_range_penalty", "compute_endpoint_penalty", "compute_monotonicity_penalty",

    "train_icrf"
]