"""
Module for collate functionality. TODO: Likely subject to refactoring before 1.0.0 release.
"""
import torch
from torch.utils.data._utils.collate import default_collate


def custom_collate(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """
    Custom collate function for handling possible None std images. If any Nones are found in the batch, the whole
    batch is set to None.
    Args:
        batch: the data batch from a Dataset as a tuple. Expects tuple of four items, similar to the return value.

    Returns:
        Batched data in a tuple
        - Index tensor containing the indices that were utilized from the dataset
        - Image tensor
        - Possible uncertainty image tensor.
        - Dictionary of metadata keys (str) and numeric values (torch.Tensor).
    """

    sorted_batch = sorted(batch, key=lambda x: x[3]['exposure_time'])
    indices, val_images, std_images, metas = zip(*sorted_batch)

    # Collate val_images and metas normally.
    index_batch = default_collate(indices)
    val_batch = default_collate(val_images)
    meta_batch = default_collate(metas)

    # Std_images are either collated normally or the batch is set to None.
    contains_none = False
    for std in std_images:
        if std is None:
            contains_none = True
            break

    if contains_none:
        std_batch = None
    else:
        std_batch = default_collate(std_images)

    return index_batch, val_batch, std_batch, meta_batch
