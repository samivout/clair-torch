"""
Module for the function that is used to linearize images as single images.
"""
from typing import Optional, Sequence, Iterable, Generator

import torch
from torch.utils.data import DataLoader

from clair_torch.common import general_functions as gf, transforms as tr
from clair_torch.datasets.image_dataset import ArtefactMapDataset
from models import ICRFModelBase


def linearize_dataset_generator(
        dataloader: DataLoader,
        device: str | torch.device,
        icrf_model: ICRFModelBase,
        flatfield_dataset: Optional[ArtefactMapDataset] = None,
        gpu_transforms: Optional[tr.Transform | Sequence[tr.Transform]] = None
) -> Generator[tuple[torch.Tensor, torch.Tensor, dict], None, None]:
    """
    Generator function to yield single linearized image and its possible associated uncertainty.
    Args:
        dataloader: Torch dataloader with custom collate function.
        device: the device on which to run the linearization.
        icrf_model: the ICRF model used to linearize the images.
        flatfield_dataset: An ArtefactMapDataset, used to select the appropriate flat field correction image for each
            linearized image.
        gpu_transforms: transform operations to run on each image as the first thing in the process.

    Returns:
        A generator object yielding a tuple of image, uncertainty image and metadata dictionary.
    """
    if not dataloader.batch_size == 1:
        raise ValueError("For linearization only batch_size of 1 is allowed.")

    if isinstance(gpu_transforms, Iterable):
        gpu_transforms = list(gpu_transforms)
    else:
        gpu_transforms = [gpu_transforms] if gpu_transforms else []

    if flatfield_dataset is not None:
        flatfield_val, flatfield_std, _ = flatfield_dataset.get_matching_artefact_image(
            dataloader.dataset.files[0], matching_attributes=["magnification, illumination"]
        )
        flatfield_val = flatfield_val.to(device=device)
        flatfield_mean = gf.flat_field_mean(flatfield_val, 1.0)
        if flatfield_std is not None:
            flatfield_std = flatfield_std.to(device=device)
        flatfield_val.requires_grad_(True)

    for i, (val_batch, std_batch, meta_batch) in enumerate(dataloader):

        # Stage tensors.
        images = val_batch.to(device)
        stds = std_batch.to(device) if std_batch is not None else None

        # Run GPU transforms
        for transform in gpu_transforms:
            images = transform(images)

        # Stage gradient usage.
        images.requires_grad_(stds is not None)

        # Linearize.
        with torch.set_grad_enabled(stds is not None):
            linearized = icrf_model(images)

        # Compute uncertainty of the linearization.
        variance = torch.zeros_like(images)
        if stds is not None:
            linearization_grad = torch.autograd.grad(
                outputs=linearized,
                inputs=images,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=False
            )[0]
            variance = (linearization_grad * stds) ** 2

        if flatfield_dataset is not None:
            linearized = gf.flatfield_correction(linearized, flatfield_val, flatfield_mean)

        if flatfield_std is not None:
            flatfield_grad = torch.autograd.grad(
                outputs=linearized,
                inputs=flatfield_val,
                grad_outputs=torch.ones_like(linearized),
                retain_graph=False
            )[0]
            variance += (flatfield_grad * flatfield_std) ** 2

        yield linearized.squeeze().detach().cpu(), torch.sqrt(variance).squeeze().detach().cpu(), meta_batch


if __name__ == "__main__":
    pass
