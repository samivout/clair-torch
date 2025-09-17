from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from typeguard import typechecked

from clair_torch.models.base import ICRFModelBase
from clair_torch.training.losses import compute_monotonicity_penalty, compute_range_penalty, compute_endpoint_penalty, \
    compute_smoothness_penalty, combined_gaussian_pair_weights, pixelwise_linearity_loss, compute_spatial_linearity_loss
from clair_torch.visualization.plotting import update_loss_plot
from clair_torch.common.general_functions import get_valid_exposure_pairs, get_pairwise_valid_pixel_mask


@typechecked
def train_icrf(
        dataloader: DataLoader,
        batch_size: int,
        device: str | torch.device,
        icrf_model: ICRFModelBase,
        optimizers: Optional[list[Optimizer]] = None,
        schedulers: Optional[list[_LRScheduler | ReduceLROnPlateau | None]] = None,
        use_relative_linearity_loss: bool = True,
        use_uncertainty_weighting: bool = True,
        epochs: int = 150,
        patience: int = 300,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        lower_valid_threshold: float = 1/255,
        upper_valid_threshold: float = 254/255,
        exposure_ratio_threshold: float = 0.1
) -> ICRFModelBase:
    """
    Training loop for the ICRF model. Requires a dataloader to yield batches of images for linearization and loss
    computation, an initialized ICRF model with a matching number of channels and pixel values.
    Args:
        dataloader: ImageDataset object for the value images.
        batch_size: the size of the batches associated with the dataloader.
        device: the device to perform the training on.
        icrf_model: initialized ICRFModel object.
        optimizers: list of PyTorch optimizers to use. Must either be a single optimizer or one for each channel.
            If None is given, then initializes default optimizers for each channel.
        schedulers: list of PyTorch learning rate schedulers. None, or list of Schedulers and Nones of equal length to
            optimizers.
        use_relative_linearity_loss: whether to use relative (True) or absolute values (False) for linearity loss
            computation.
        use_uncertainty_weighting: whether to utilize inverse uncertainty based weights in loss computation.
        epochs: number of epochs to train the model.
        patience: number of epochs until early stopping when loss does not improve.
        alpha: coefficient for monotonicity penalty term.
        beta: coefficient for range penalty term.
        gamma: coefficient for endpoint penalty term.
        delta: coefficient for smoothness penalty term.
        lower_valid_threshold: lower exclusive bound for considering a pixel as valid.
        upper_valid_threshold: upper exclusive bound for considering a pixel as valid.
        exposure_ratio_threshold: threshold for rejecting image pairs based on the ratio of the shorter exposure time
            against the longer exposure time of a pair of images. Should be between [0.0, 1.0]. Pairs with values lower
            than the threshold value are rejected from the linearity loss computation.

    Returns:
        The trained model.
    """
    channels = icrf_model.channels

    if batch_size == 1:
        raise ValueError("Batch size must be larger than 1.")

    if optimizers is None:
        optimizers = [
            torch.optim.Adam(icrf_model.channel_params(c), lr=1e-3, amsgrad=False) for c in range(channels)
        ]

    previous_lrs = [pg['lr'] for opt in optimizers for pg in opt.param_groups]

    if schedulers is None:
        schedulers = [None] * len(optimizers)
    if len(schedulers) != len(optimizers):
        raise ValueError(f"Mismatched number of optimizers: {len(optimizers)} and schedulers: {len(schedulers)}.")

    best_losses = [float('inf')] * channels
    epochs_without_improvement = [0] * channels

    icrf_model.train()
    icrf_model.plot_icrf()

    for epoch in range(epochs):

        running_loss = torch.zeros(icrf_model.channels, device=icrf_model.icrf.get_device())

        for index_batch, val_batch, std_batch, meta_batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", total=batch_size):

            images = val_batch.to(device=device)
            stds = std_batch.to(device) if std_batch is not None else None
            exposures = meta_batch['exposure_time'].to(device=device)

            # Skip batch if number of images in batch is only one.
            if images.shape[0] < 2:
                print("Skipped batch due to single image.")
                continue

            i_idx, j_idx, ratio_pairs = get_valid_exposure_pairs(increasing_exposure_values=exposures,
                                                                 exposure_ratio_threshold=exposure_ratio_threshold)
            valid_mask = get_pairwise_valid_pixel_mask(images, i_idx, j_idx, stds,
                                                       val_lower=lower_valid_threshold, val_upper=upper_valid_threshold)
            gaussian_weight = combined_gaussian_pair_weights(images, i_idx, j_idx)

            for optimizer in optimizers:
                optimizer.zero_grad()

            images.requires_grad_(True)
            linearized = icrf_model(images)  # Shape: (N, C, H, W)

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

            icrf_curve = icrf_model.icrf

            pixelwise_loss, pixelwise_errors = pixelwise_linearity_loss(linearized, i_idx, j_idx, ratio_pairs,
                                                                        linearized_stds, use_relative_linearity_loss)

            spatial_linearity_loss, _, _ = (
                compute_spatial_linearity_loss(pixelwise_loss, pixelwise_errors, gaussian_weight, valid_mask, use_uncertainty_weighting))

            linearity_loss = torch.sqrt((spatial_linearity_loss ** 2).sum(dim=0))

            monotonicity_loss = compute_monotonicity_penalty(icrf_curve, per_channel=True)
            range_loss = compute_range_penalty(icrf_curve, per_channel=True)
            endpoint_loss = compute_endpoint_penalty(icrf_curve, per_channel=True)
            smoothness_loss = compute_smoothness_penalty(icrf_curve, per_channel=True)

            loss = linearity_loss + alpha * monotonicity_loss + beta * range_loss + gamma * endpoint_loss + delta * smoothness_loss

            if len(optimizers) == 1:
                loss = torch.sum(loss)

            for c, optimizer in enumerate(optimizers):
                loss[c].backward(retain_graph=True)

            for c, optimizer in enumerate(optimizers):
                optimizer.step()

            icrf_model.update_icrf()

            running_loss += loss.detach()

        avg_loss = (running_loss / len(dataloader)).cpu().numpy()
        print(f"Epoch {epoch + 1} Loss: {avg_loss}")
        update_loss_plot(epoch, avg_loss)

        for c in range(channels):
            if avg_loss[c] < best_losses[c]:
                best_losses[c] = avg_loss[c]
                epochs_without_improvement[c] = 0
            else:
                epochs_without_improvement[c] += 1

        if all(epochs_without_improvement[c] >= patience for c in range(icrf_model.channels)):
            print(f"Early stopping triggered for all channels (patience = {patience} epochs).")
            break

        for c, scheduler in enumerate(schedulers):
            if scheduler is not None:
                scheduler.step(avg_loss[c])

        for i, optimizer in enumerate(optimizers):
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != previous_lrs[i]:
                print(f"Optimizer {i} learning rate changed to: {current_lr}")
            previous_lrs[i] = current_lr

        if (epoch + 1) % 5 == 0:
            icrf_model.plot_icrf()

    return icrf_model


if __name__ == "__main__":
    pass
