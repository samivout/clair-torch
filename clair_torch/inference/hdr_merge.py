"""
Module for the function that is used to merge a set of images into a single HDR image.
"""
from typing import Optional, Callable, Sequence, Iterable

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from clair_torch.common import general_functions as gf, transforms as tr
from clair_torch.datasets.image_dataset import FlatFieldArtefactMapDataset, DarkFieldArtefactMapDataset, ImageMapDataset
from clair_torch.models.base import ICRFModelBase
from clair_torch.training.losses import gaussian_value_weights
from clair_torch.common.statistics import WBOMean


def compute_hdr_image(dataloader: DataLoader, device: str | torch.device,
                      icrf_model: Optional[ICRFModelBase] = None, weight_fn: Optional[Callable] = None,
                      flat_field_dataset: Optional[FlatFieldArtefactMapDataset] = None,
                      gpu_transforms: Optional[tr.BaseTransform | Sequence[tr.BaseTransform]] = None,
                      dark_field_dataset: Optional[DarkFieldArtefactMapDataset] = None) \
        -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Function for computing HDR merging of a set of images at different exposure times under stationary conditions.
    Uncertainty can be computed if the dataset is provided with PairedFrameSettings instances that include a path
    to an uncertainty image. Flat field correction can be performed if an FlatFieldArtefactMapDataset is provided and a
    matching artefact image is found. Similarly, a dark field correction can be performed with a
    DarkFieldArtefactMapDataset.
    Args:
        dataloader: DataLoader containing an ImageMapDataset instance, which should contain FrameSettings or
            PairedFrameSettings instances.
        device: the device to run the computations on.
        icrf_model: an optional ICRF model to use to linearize the images before merging.
        weight_fn: a weighting function that takes as input the batch of images.
        flat_field_dataset: a FlatFieldArtefactMapDataset for flat field correction.
        gpu_transforms: Optional transform operations to be performed on the image batch after moving the data to the
            desired device.
        dark_field_dataset: a DarkFieldArtefactMapDataset for dark field correction.

    Returns:
        A tuple of tensors representing (the HDR image, uncertainty of the HDR image or None).
    """
    expected_number_of_iterations = len(dataloader)
    main_dataset: ImageMapDataset = dataloader.dataset

    if isinstance(gpu_transforms, Iterable):
        gpu_transforms = gpu_transforms
    else:
        gpu_transforms = [gpu_transforms]

    running_average = None
    running_variance = None

    mean_handler = WBOMean(dim=0)

    # HDR val and std image computations.
    for index_batch, val_batch, std_batch, meta_batch in tqdm(dataloader, desc="Batches processed", total=expected_number_of_iterations):

        # Stage tensors on correct device.
        images = val_batch.to(device=device)
        stds = std_batch.to(device) if std_batch is not None else None
        exposures = meta_batch['exposure_time'].to(device=device)

        # Run GPU transforms.
        for transform in gpu_transforms:
            if transform is not None:
                images = transform(images)

        # Set gradient requirements.
        images.requires_grad_(stds is not None)

        dark_field_val, dark_field_std = None, None
        if dark_field_dataset is not None:

            frame_settings_in_batch = []
            for index in index_batch:
                frame_settings_in_batch.append(main_dataset.files[index])

            _, dark_field_val, dark_field_std, _ = dark_field_dataset.get_matching_artefact_images(frame_settings_in_batch)

            if dark_field_val is not None:
                dark_field_val, dark_field_std = dark_field_val.to(device=device), dark_field_std.to(device=device)

                dark_field_val.requires_grad_(dark_field_std is not None)

                with torch.set_grad_enabled(dark_field_std is not None):
                    images = gf.conditional_gaussian_blur(images, dark_field_val, threshold=0.05, kernel_size=3,
                                                          differentiable=True)

        # Compute weights
        batch_weight = torch.ones_like(images) if weight_fn is None else gaussian_value_weights(images)

        exposures_view = exposures.view(-1, 1, 1, 1)

        def linearize(imgs):
            return icrf_model(imgs) if icrf_model else imgs

        with torch.set_grad_enabled(stds is not None):
            linearized = linearize(images) / exposures_view

        running_average = mean_handler.update_values(linearized, batch_weight)

        if stds is not None:
            running_gradient = torch.autograd.grad(
                outputs=running_average,
                inputs=images,
                grad_outputs=torch.ones_like(running_average),
                retain_graph=True
            )[0]
            variance_update = torch.sum((running_gradient * stds) ** 2, dim=0, keepdim=True)
            running_variance = variance_update if running_variance is None else running_variance + variance_update

        if dark_field_std is not None:
            running_gradient = torch.autograd.grad(
                outputs=running_average,
                inputs=dark_field_val,
                grad_outputs=torch.ones_like(running_average),
                retain_graph=False
            )[0]

            variance_update = torch.sum((running_gradient * dark_field_std) ** 2, dim=0, keepdim=True)
            running_variance = running_variance + variance_update

        mean_handler.internal_detach()

    # Flatfield corrections.
    if flat_field_dataset is not None:

        _, flatfield_val, flatfield_std, _ = flat_field_dataset.get_matching_artefact_images([main_dataset.files[0]])
        flatfield_val, flatfield_std = flatfield_val.to(device=device), flatfield_std.to(device=device)

        if flatfield_std is not None:
            flatfield_val.requires_grad_(True)

        flatfield_mean = gf.flat_field_mean(flatfield_val, 1.0)

        flatfield_corrected_running_average = gf.flatfield_correction(running_average, flatfield_val, flatfield_mean)

        if flatfield_std is not None:
            running_gradient = torch.autograd.grad(
                outputs=flatfield_corrected_running_average,
                inputs=flatfield_val,
                grad_outputs=torch.ones_like(flatfield_corrected_running_average),
                retain_graph=False
            )[0]

            running_variance = running_variance + (running_gradient * flatfield_std) ** 2

        running_average = flatfield_corrected_running_average

    return running_average.squeeze(), torch.sqrt(running_variance.squeeze()) if running_variance is not None else None


if __name__ == "__main__":
    pass
