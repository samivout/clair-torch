"""This module contains all the enums used in this project.

The enums are used to manage clear definitions of different kinds of modes some functions expose.
"""

import torch
from enum import Enum, auto


class InterpMode(Enum):
    """
    Manages the interpolation modes used in ICRF model classes.
    """
    LOOKUP = auto()  # nearest-neighbour LUT, no-grad fast path
    LINEAR = auto()
    CATMULL = auto()


class DarkFieldMode(Enum):
    MAX = auto()
    INTERPOLATED = auto()
    CLOSEST = auto()
    ERROR = auto()
    SKIP_BATCH = auto()


class FlatFieldMode(Enum):
    CLOSEST = auto()
    ERROR = auto()
    SKIP_BATCH = auto()


class MissingStdMode(Enum):
    """
    Manages how missing uncertainty images are dealt with in ImageDataset classes.
    """
    NONE = auto()
    CONSTANT = auto()
    MULTIPLIER = auto()


class MissingValMode(Enum):
    """
    Manages how missing value images are dealt with in ImageDataset classes.
    """
    ERROR = auto()
    SKIP_BATCH = auto()


class ChannelOrder(Enum):
    RGB = auto()
    BGR = auto()
    ANY = auto()


class DimensionOrder(Enum):
    """
    Manges the ordering of data dimensions.
    """
    BCS = auto()    # Batch, Channel, Spatial - Common for PyTorch and ML packages.
    BSC = auto()    # Batch, Spatial, Channel - common for OpenCV


class VarianceMode(Enum):
    """
    Manages how the variance is computed in the WBOMeanVar class.
    """
    POPULATION = auto()
    SAMPLE_FREQUENCY = auto()
    RELIABILITY_WEIGHTS = auto()


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

REVERSE_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

