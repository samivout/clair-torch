"""This module contains all the enums used in this project.

The enums are used to manage clear definitions of different kinds of modes some functions expose.
"""

from enum import Enum, auto


class InterpMode(Enum):
    """
    Manages the interpolation modes used in ICRF model classes.
    """
    LOOKUP = auto()  # nearest-neighbour LUT, no-grad fast path
    LINEAR = auto()
    CATMULL = auto()


class MissingStdMode(Enum):
    """
    Manages how missing uncertainty images are dealt with in ImageDataset classes.
    """
    NONE = auto()
    CONSTANT = auto()
    MULTIPLIER = auto()


class ChannelOrder(Enum):
    RGB = auto()
    BGR = auto()
    ANY = auto()


class VarianceMode(Enum):
    """
    Manages how the variance is computed in the WBOMeanVar class.
    """
    POPULATION = auto()
    SAMPLE_FREQUENCY = auto()
    RELIABILITY_WEIGHTS = auto()

