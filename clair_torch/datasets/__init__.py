"""
The datasets subpackage provides dataset classes, which can be used with the PyTorch Dataloader class for managing the
data loading process in functions.
"""

from .base import MultiFileMapDataset, MultiFileIterDataset
from .collate import custom_collate
from .image_dataset import ImageMapDataset, FlatFieldArtefactMapDataset
from .video_frame_dataset import VideoIterableDataset

__all__ = [
    "MultiFileMapDataset", "MultiFileIterDataset",

    "custom_collate",

    "ImageMapDataset", "FlatFieldArtefactMapDataset",

    "VideoIterableDataset"
]
