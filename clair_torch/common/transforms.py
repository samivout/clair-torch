"""
Module for transforms to be used with PyTorch tensors. Main use is for chaining required operations on in __getitem__
of ImageDataset class.
"""
from typing import Protocol, Optional, Tuple, Any, Type, runtime_checkable
from abc import ABC, abstractmethod

import torch

from clair_torch.common.enums import DTYPE_MAP, REVERSE_DTYPE_MAP
from clair_torch.common.general_functions import (cv_to_torch, torch_to_cv, normalize_tensor, clamp_along_dims,
                                                  normalize_container)
from clair_torch.validation.type_checks import validate_all


TRANSFORM_REGISTRY: dict[str, Type] = {}


def register_transform(cls):
    """
    Decorator function to build a registry of defined Transform classes. The registry is used in conjunction
    with the serialization and deserialization functions.
    Args:
        cls: the class to be registered.

    Returns:
        The input class.
    """
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


class BaseTransform(ABC):
    """
    Base class for Transform classes, which typically wrap a function from general functions. Must be callable, taking
    a torch.Tensor as input and return a torch.Tensor.
    """
    def __init__(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """
        Serialization function, which dumps the instance into a dictionary representation.
        Returns: dictionary that can be used to deserialize back into an equivalent instance.
        """
        ...

    @classmethod
    def from_config(cls, cfg: dict[str, Any]):
        """
        Deserialization function, which initializes a new instance from a dictionary representation
        of the class.
        Args:
            cfg: the configuration dictionary.

        Returns:
            A new instance of this class.
        """
        return cls(**cfg)


class TensorParams:
    """
    A mix-in class for inheriting an easy method to move tensors from one device to another.
    """
    def __init__(self, tensor_attributes: list[str]):
        self.tensor_attributes = tensor_attributes

    def to_device(self, device: str | torch.device):
        for attr in self.tensor_attributes:
            if hasattr(self, attr) and self.__getattribute__(attr) is not None:
                self.__getattribute__(attr).to(device=device)


@runtime_checkable
class Transform(Protocol):
    """
    Class stub for typing purposes. Used to indicate that a transform such as the ones in this module is required.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


@register_transform
class CvToTorch(BaseTransform):
    """
    Transform for modifying a tensor from OpenCV dimensions and channel ordering to a PyTorch dimensionality and
    ordering.
    """

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return cv_to_torch(x)

    def to_config(self) -> dict[str, Any]:
        return {}


@register_transform
class TorchToCv(BaseTransform):
    """
    Transform for converting a tensor from PyTorch (C, H, W) format with RGB channels to OpenCV format (H, W, C) with
    BGR channels.
    """

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return torch_to_cv(x)

    def to_config(self) -> dict[str, Any]:
        return {}


@register_transform
class Normalize(BaseTransform):
    """
    Transform for normalizing a tensor by a minimum and maximum value.
    If no values are given, dynamically use min and max of the given tensor.
    """
    def __init__(
        self,
        max_val: Optional[float] = None,
        min_val: Optional[float] = None,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        target_range = normalize_container(target_range, tuple, convert_if_iterable=True)
        validate_all([max_val, min_val], (float, int), raise_error=True, allow_none_elements=True)
        validate_all(target_range, float, raise_error=True, allow_none_elements=False)

        self.max_val = max_val
        self.min_val = min_val
        self.target_range = target_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(x, max_val=self.max_val, min_val=self.min_val, target_range=self.target_range)

    def to_config(self) -> dict[str, Any]:
        return {
            "max_val": self.max_val,
            "min_val": self.min_val,
            "target_range": list(self.target_range),
        }


@register_transform
class ClampAlongDims(BaseTransform):
    """
    Transform for clamping the tensor values between a min and max value, along the given dimension(s).
    """
    def __init__(self, dim: int | tuple[int, ...], min_max_pairs: tuple[float, float] | list[tuple[float, float]]):

        super().__init__()
        self.dim = dim
        self.min_max_pairs = min_max_pairs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return clamp_along_dims(x, self.dim, self.min_max_pairs)

    def to_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "min_max_pairs": self.min_max_pairs,
        }


@register_transform
class CastTo(BaseTransform):
    """
    Transform for casting the tensor to the given datatype and device. If data_type or device are not give, then
    maintain them as they are.
    """
    def __init__(self, data_type: Optional[str | torch.dtype] = None, device: Optional[str | torch.device] = None):
        super().__init__()
        if isinstance(data_type, str):
            data_type = DTYPE_MAP[data_type]
        if isinstance(device, str):
            device = torch.device(device)
        validate_all([data_type], torch.dtype, raise_error=True, allow_none_elements=True)
        validate_all([device], (str, torch.device), raise_error=True, allow_none_elements=True)

        self.data_type = data_type
        self.device = device

    def __call__(self, x: torch.Tensor):

        data_type = self.data_type if self.data_type is not None else x.dtype
        device = self.device if self.device is not None else x.device

        return x.to(dtype=data_type, device=device)

    def to_config(self) -> dict[str, Any]:
        dtype = REVERSE_DTYPE_MAP[self.data_type]
        return {
            "data_type": dtype,
            "device": str(self.device) if self.device is not None else None
        }


@register_transform
class StridedDownscale(BaseTransform):
    """
    Transform for applying a spatial downscale on the input tensor.
    """
    def __init__(self, step_size: int):
        super().__init__()
        validate_all([step_size], int, raise_error=True, allow_none_elements=False)
        if step_size < 0:
            raise ValueError(f"step_size must be non-negative.")

        self.step_size = step_size

    def __call__(self, x: torch.Tensor):

        strided = x[..., ::self.step_size, ::self.step_size]

        return strided

    def to_config(self) -> dict[str, Any]:
        return {
            "step_size": self.step_size
        }


def serialize_transforms(transforms: list) -> list[dict[str, Any]]:
    """
    Utility function for serializing multiple transforms. This should be used as the main method of serializing
    transforms.
    Args:
        transforms: a list of Transform instances.

    Returns:
        A list of dictionaries, each representing a serialized transform.
    """
    return [
        {"type": t.__class__.__name__, "params": t.to_config()} for t in transforms
    ]


def deserialize_transforms(cfg_list: list[dict[str, Any]]):
    """
    Utility function to deserialize transforms. This should be used as the main method of deserializing Transforms.
    Args:
        cfg_list: a list of dictionaries, representing serialized Transforms.

    Returns:
        A list of deserialized Transforms.
    """
    result = []
    for cfg in cfg_list:
        cls = TRANSFORM_REGISTRY[cfg["type"]]
        result.append(cls.from_config(cfg["params"]))
    return result


if __name__ == "__main__":
    pass
