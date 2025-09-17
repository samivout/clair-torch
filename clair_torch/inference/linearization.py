"""
Module for the function that is used to linearize single images.
"""
from typing import Optional, Sequence, Iterable, Generator

import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from clair_torch.common import general_functions as gf, transforms as tr
from clair_torch.datasets.image_dataset import FlatFieldArtefactMapDataset, DarkFieldArtefactMapDataset, ImageMapDataset
from clair_torch.models.base import ICRFModelBase


@typechecked
def linearize_dataset_generator(
        dataloader: DataLoader,
        device: str | torch.device,
        icrf_model: ICRFModelBase,
        flatfield_dataset: Optional[FlatFieldArtefactMapDataset] = None,
        gpu_transforms: Optional[tr.Transform | Sequence[tr.Transform]] = None,
        dark_field_dataset: Optional[DarkFieldArtefactMapDataset] = None
) -> Generator[tuple[torch.Tensor, torch.Tensor, dict], None, None]:
    """
    Generator function to yield single linearized image and its possible associated uncertainty.
    Args:
        dataloader: Torch dataloader with custom collate function.
        device: the device on which to run the linearization.
        icrf_model: the ICRF model used to linearize the images.
        flatfield_dataset: An FlatFieldArtefactMapDataset, used to select the appropriate flat field correction image
            for each linearized image.
        gpu_transforms: transform operations to run on each image as the first thing in the process.
        dark_field_dataset: An DarkFieldArtefactMapDataset, used to select the appropriate dark field correction image
            for each linearized image.

    Returns:
        A generator object yielding a tuple of image, uncertainty image and metadata dictionary.
    """
    main_dataset: ImageMapDataset = dataloader.dataset

    if not dataloader.batch_size == 1:
        raise ValueError("For linearization only batch_size of 1 is allowed.")

    if isinstance(gpu_transforms, Iterable):
        gpu_transforms = list(gpu_transforms)
    else:
        gpu_transforms = [gpu_transforms] if gpu_transforms else []

    flatfield_val, flatfield_std = None, None
    if flatfield_dataset is not None:

        _, flatfield_val, flatfield_std, _ = flatfield_dataset.get_matching_artefact_images([main_dataset.files[0]])
        flatfield_val = flatfield_val.to(device=device)
        flatfield_mean = gf.flat_field_mean(flatfield_val, 1.0)
        flatfield_std = flatfield_std.to(device=device) if flatfield_std is not None else None
        flatfield_val.requires_grad_(True)

    for i, (index_batch, val_batch, std_batch, meta_batch) in enumerate(dataloader):

        # Stage tensors.
        images = val_batch.to(device)
        stds = std_batch.to(device) if std_batch is not None else None

        # Run GPU transforms
        for transform in gpu_transforms:
            if transform is not None:
                images = transform(images)

        # Stage gradient usage.
        images.requires_grad_(stds is not None)

        # Apply dark field correction.
        dark_field_val, dark_field_std = None, None
        if dark_field_dataset is not None:

            frame_settings_in_batch = []
            for index in index_batch:
                frame_settings_in_batch.append(main_dataset.files[index])

            _, dark_field_val, dark_field_std, _ = dark_field_dataset.get_matching_artefact_images(
                frame_settings_in_batch)

            if dark_field_val is not None:
                dark_field_val, dark_field_std = dark_field_val.to(device=device), dark_field_std.to(device=device)

                dark_field_val.requires_grad_(dark_field_std is not None)

                with torch.set_grad_enabled(dark_field_std is not None):
                    images = gf.conditional_gaussian_blur(images, dark_field_val, threshold=0.05, kernel_size=3,
                                                          differentiable=True)

        # Linearize.
        with torch.set_grad_enabled(stds is not None):
            linearized = icrf_model(images)

        # Compute uncertainty of the linearization.
        running_variance = torch.zeros_like(images)
        if stds is not None:
            running_gradient = torch.autograd.grad(
                outputs=linearized,
                inputs=images,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=True
            )[0]
            running_variance = running_variance + (running_gradient * stds) ** 2

        # Compute uncertainty of the dark field correction.
        if dark_field_std is not None:
            running_gradient = torch.autograd.grad(
                outputs=linearized,
                inputs=dark_field_val,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=False
            )[0]
            running_variance = running_variance + (running_gradient * dark_field_std) ** 2

        # Apply flat field correction to the linearized images.
        if flatfield_dataset is not None:
            linearized = gf.flatfield_correction(linearized, flatfield_val, flatfield_mean)

        # Compute the uncertainty of the flat field correction.
        if flatfield_std is not None:
            running_gradient = torch.autograd.grad(
                outputs=linearized,
                inputs=flatfield_val,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=False
            )[0]
            running_variance = running_variance + (running_gradient * flatfield_std) ** 2

        yield linearized.squeeze().detach().cpu(), torch.sqrt(running_variance).squeeze().detach().cpu(), meta_batch


if __name__ == "__main__":
    pass
