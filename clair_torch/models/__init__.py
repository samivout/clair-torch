"""
The models package provides the inverse camera response function model classes, which are built on the torch.nn.module
class. The base model provides a guideline for creating a new ICRF model, while the concrete implementations provide
different approaches to modelling an ICRF.
"""
from .base import ICRFModelBase
from .icrf_model import ICRFModelDirect, ICRFModelPCA

__all__ = [
    "ICRFModelBase", "ICRFModelPCA", "ICRFModelDirect"
]
