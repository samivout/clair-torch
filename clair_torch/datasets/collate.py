"""
Module for collate functionality. Likely subject to refactoring before 1.0.0 release.
"""
from torch.utils.data._utils.collate import default_collate


def custom_collate(batch):
    """
    Custom collate function for handling possible None std images. If any Nones are found in the batch, the whole
    batch is set to None.
    Args:
        batch:

    Returns:

    """

    sorted_batch = sorted(batch, key=lambda x: x[2]['exposure_time'])
    val_images, std_images, metas = zip(*sorted_batch)

    # Collate val_images and metas normally.
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

    return val_batch, std_batch, meta_batch
