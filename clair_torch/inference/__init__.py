"""
The inference subpackage provides functionality that can be used to measure a camera's linearity, create HDR images,
linearize single images and compute quantitatively well-defined mean and uncertainty images from a stack of images or
a video.
"""

from .hdr_merge import compute_hdr_image
from .inferential_statistics import compute_video_mean_and_std
from .linearization import linearize_dataset_generator
from .measure_linearity import measure_linearity

__all__ = [
    "compute_hdr_image",

    "compute_video_mean_and_std",

    "linearize_dataset_generator",

    "measure_linearity"
]
