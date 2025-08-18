"""
The metadata subpackage provides classes for managing information related to the files, which are managed by the
FrameData and PairedFrameData classes. The BaseMetadata class provides a guideline for implementing a Metadata class,
while the others provide concrete ready-to-use classes to manage image metadata and video metadata.
"""

from .base import BaseMetadata
from .imaging_metadata import ImagingMetadata, VideoMetadata

__all__ = [
    "BaseMetadata", "ImagingMetadata", "VideoMetadata"
]

