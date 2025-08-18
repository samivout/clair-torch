from abc import ABC, abstractmethod
from typing import Sequence, Type


class BaseMetadata(ABC):
    """
    Base class for enforcing the format of metadata classes. Each metadata class must implement their
    own __init__ method and have a common interface for checking metadata matches via is_match.
    """
    @abstractmethod
    def __init__(self):
        ...

    @property
    @abstractmethod
    def _numeric_fields(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def _text_fields(self) -> list[str]:
        ...

    def get_numeric_metadata(self) -> dict[str, float | int]:
        return {
            field: getattr(self, field)
            for field in self._numeric_fields
            if getattr(self, field) is not None
        }

    def get_text_metadata(self) -> dict[str, str]:
        return {
            field: getattr(self, field)
            for field in self._text_fields
            if getattr(self, field) is not None
        }

    def get_all_metadata(self) -> dict[str, str | int | float]:
        numeric_dict = self.get_numeric_metadata()
        text_dict = self.get_text_metadata()
        return text_dict | numeric_dict

    def is_match(self, other: Type['BaseMetadata'], attributes: Sequence[str]) -> bool:
        """
        Function for checking if two instances of metadata classes are a match or not based on the given
        sequence of attributes.
        Args:
            other: an instance of a concrete implementation of a metadata class.
            attributes: sequence of attributes used to determine if the two instances are a match.

        Returns:
            True if a match, False if not.
        """
        if not issubclass(type(other), BaseMetadata):
            return False
        for attr in attributes:
            if getattr(self, attr, None) != getattr(other, attr, None):
                return False
        return True
