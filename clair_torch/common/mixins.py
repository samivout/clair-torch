"""
A module for mixin classes to be used via (multiple) inheritance with other classes.
"""
import copy
from typing import Any


class Clearable:
    """
    A mixin class to define certain attributes, whose values are easy to clear or set to the given default on call.
    """
    def __init__(self, clearable_attributes: dict[str, Any]):
        self.clearable_attributes = clearable_attributes

    def clear(self) -> None:
        """
        Method for resetting the clearable attributes to the given default value on call.
        """
        for attr, default in self.clearable_attributes.items():

            if callable(default):
                value = default()
            else:
                value = copy.deepcopy(default)

            setattr(self, attr, value)
