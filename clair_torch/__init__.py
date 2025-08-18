"""
The top-level package of clair_torch. Currently, directly exposes the functionality that each subpackage exposes in
turn.
"""
from . import common, datasets, inference, metadata, models, training, validation, visualization

__version__ = "0.1.0"

__all__ = [
    "common", "datasets", "inference", "metadata", "models", "training", "validation", "visualization"
]
