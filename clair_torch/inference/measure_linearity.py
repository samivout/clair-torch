"""
Module for the function that can be used to measure and visualize the linearity of a camera.
"""
from typing import Optional

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from clair_torch.training.losses import combined_gaussian_pair_weights, pixelwise_linearity_loss, compute_spatial_linearity_loss
from clair_torch.common.general_functions import get_valid_exposure_pairs, get_pairwise_valid_pixel_mask
from clair_torch.models.base import ICRFModelBase


@typechecked
def measure_linearity(dataloader: DataLoader, device: str | torch.device,
                      use_uncertainty_weighting: bool = True, use_relative_linearity_loss: bool = True,
                      icrf_model: Optional[ICRFModelBase] = None) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Function for measuring the linearity of a batch of images by computing the spatial means of the pixelwise linearity
    losses of pairs of images captured at different exposure times, but otherwise stationary conditions. Returns for
    each pair of images the exposure time ratios, spatial linearity losses, the standard deviation of the spatial
    linearity losses, and the uncertainty of the spatial linearity losses. By leaving the ICRF model None, you can
    get a measure of the camera's linearity as it is on its own.
    Args:
        dataloader: an ImageMapDataset consisting of FrameSettings or PairedFrameSettings objects.
        device: the device to run the computation on.
        use_uncertainty_weighting: whether to utilize inverse uncertainty weights or not.
        use_relative_linearity_loss: whether to utilize relative or absolute values for linearity loss computation.
        icrf_model: an optional ICRF model to utilize for linearization of images.

    Returns:
        A tuple of tensors representing (exposure time ratios, spatial linearity losses,
        standard deviation of spatial linearity losses, uncertainty of the spatial linearity losses).
    """

    for index_batch, val_batch, std_batch, meta_batch in tqdm(dataloader, desc="Creating linearity plot"):

        images = val_batch.to(device=device)
        stds = std_batch.to(device=device) if std_batch is not None else None
        exposures = meta_batch['exposure_time'].to(device=device)

        i_idx, j_idx, ratio_pairs = get_valid_exposure_pairs(increasing_exposure_values=exposures, exposure_ratio_threshold=0.2)
        valid_mask = get_pairwise_valid_pixel_mask(images, i_idx, j_idx, stds, val_lower=1 / 255, val_upper=254 / 255)
        gaussian_weight = combined_gaussian_pair_weights(images, i_idx, j_idx)

        # Set gradient requirements.
        images.requires_grad_(stds is not None)

        # Linearize if an ICRF model is provided.
        linearized = icrf_model(images) if icrf_model is not None else images

        # Compute uncertainty of linearization if uncertainty images are available.
        if stds is not None:
            grads = torch.autograd.grad(
                outputs=linearized,
                inputs=images,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=True
            )[0]
            linearized_stds = (grads * stds).abs()
        else:
            linearized_stds = None

        pixelwise_loss, pixelwise_errors = pixelwise_linearity_loss(linearized, i_idx, j_idx, ratio_pairs,
                                                                    linearized_stds, use_relative_linearity_loss)

        spatial_linearity_loss, spatial_linearity_loss_std, spatial_linearity_loss_error = (
            compute_spatial_linearity_loss(pixelwise_loss, pixelwise_errors, gaussian_weight, valid_mask,
                                           use_uncertainty_weighting))

        return ratio_pairs, spatial_linearity_loss, spatial_linearity_loss_std, spatial_linearity_loss_error
