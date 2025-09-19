"""
The common package provides typical utilities used with all the other packages in this project, including
    - Classes and functions for IO operations.
    - Enums used with various classes.
    - General mathematical functions and transformation operations in both function and Class forms.
"""

from .base import BaseFileSettings
from .data_io import load_image, load_icrf_txt, load_principal_components, save_icrf_txt, save_image, load_video_frames_generator
from .enums import InterpMode, MissingStdMode, ChannelOrder, VarianceMode
from .file_settings import FileSettings, PairedFileSettings, FrameSettings, PairedFrameSettings, file_settings_constructor, group_frame_settings_by_attributes
from .general_functions import weighted_mean_and_std, flat_field_mean, flatfield_correction, get_valid_exposure_pairs, get_pairwise_valid_pixel_mask, cv_to_torch, torch_to_cv, normalize_tensor, clamp_along_dims, conditional_gaussian_blur, cli_parse_args_from_config
from .statistics import WBOMean, WBOMeanVar
from .transforms import BaseTransform, CvToTorch, TorchToCv, CastTo, ClampAlongDims, Normalize, StridedDownscale

__all__ = [
    "BaseFileSettings",

    "load_image", "load_icrf_txt", "load_principal_components", "load_video_frames_generator", "save_image",
    "save_icrf_txt",

    "InterpMode", "MissingStdMode", "ChannelOrder", "VarianceMode",

    "FileSettings", "PairedFileSettings", "FrameSettings", "PairedFrameSettings", "file_settings_constructor",
    "group_frame_settings_by_attributes",

    "weighted_mean_and_std", "flat_field_mean", "flatfield_correction", "get_valid_exposure_pairs",
    "get_pairwise_valid_pixel_mask", "cv_to_torch", "torch_to_cv", "normalize_tensor", "clamp_along_dims",
    "conditional_gaussian_blur", "cli_parse_args_from_config",

    "WBOMean", "WBOMeanVar",

    "BaseTransform", "CvToTorch", "TorchToCv", "CastTo", "ClampAlongDims", "Normalize",
    "StridedDownscale"
]
