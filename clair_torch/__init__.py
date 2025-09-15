"""
The top-level package of clair_torch. Currently, directly exposes the functionality that each subpackage exposes in
turn.
"""
from . import common, datasets, inference, metadata, models, training, validation, visualization

try:
    from ._version import version as __version__
except ImportError:  # Fallback if _version.py doesn't exist yet
    __version__ = "0.0.0+unknown"

__all__ = [
    "common", "datasets", "inference", "metadata", "models", "training", "validation", "visualization"
]
