from abc import ABC, abstractmethod
from typing import Type, Optional

from math import isclose

from typeguard import typechecked


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

    @typechecked
    def is_match(self, other: 'BaseMetadata', attributes: dict[str, None | int | float], *,
                 missing_key_fails: bool = True) -> bool:
        """
        Function for checking if two instances of metadata classes are a match or not based on the given
        mapping of attributes to tolerances.

        Args:
            other: an instance of a concrete implementation of a metadata class.
            attributes: dictionary mapping attribute names to their tolerances.
                        A None value means an exact match is required.
            missing_key_fails: whether a missing attribute results in a failed check or not.

        Returns:
            True if a match, False if not.
        """
        if not issubclass(type(other), BaseMetadata):
            return False

        for attr, tolerance in attributes.items():
            if attr in self.get_text_metadata():
                if getattr(self, attr, None) != getattr(other, attr, None):
                    return False

            elif attr in self.get_numeric_metadata():
                safe_tolerance = 0.0 if tolerance is None else tolerance
                if not isclose(getattr(self, attr, None), getattr(other, attr, None), rel_tol=safe_tolerance):
                    return False

            elif missing_key_fails:
                # Attribute not present in either text or numeric metadata
                return False

        return True
