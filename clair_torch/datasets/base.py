"""
Module for the base classes of the datasets subpackage.
"""
from typing import TypeVar, List
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, IterableDataset

from clair_torch.common.file_settings import FileSettings


T_File = TypeVar("T_File", bound=FileSettings)


class MultiFileMapDataset(Dataset[T_File], ABC):
    """
    A generic base class for map-style Dataset classes. Dataset classes must manage files via a concrete implementation
    of the generic base FileSettings class.
    """
    def __init__(self, files: List[T_File]):

        self.files = files

    @abstractmethod
    def __len__(self):
        pass


class MultiFileIterDataset(IterableDataset[T_File], ABC):
    """
    A generic base class for iterable-style Dataset classes. Dataset classes must manage files via a concrete
    implementation of the generic base FileSettings class.
    """

    def __init__(self, files: List[T_File]):

        self.files = files

    @abstractmethod
    def __iter__(self):
        pass
