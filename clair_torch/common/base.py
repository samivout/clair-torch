"""
Module for the base classes of the common subpackage.
"""

from typing import Optional, List, Sequence
from pathlib import Path
from abc import ABC, abstractmethod

from typeguard import typechecked

from clair_torch.common.transforms import Transform
from clair_torch.validation.io_checks import validate_input_file_path, is_potentially_valid_file_path


class BaseFileSettings(ABC):
    """
    Base class for implementing classes that manage IO operations.

    BaseFileSettings acts as the guideline for designing classes for camera_linearity_torch's IO operations.

    Attributes
    ----------
    input_path: Path
        a filepath from which the data will be read.
    output_path: Path
        a filepath to which data can be saved. Based on the optional output_path init argument. If None is given, then
        the parent directory of input_path is used as a root in which a new directory called based on default_output_root
        is created, in which the same filename as the input_path's name will be used to create a new file upon saving.
    default_output_root: Path
        a default dirpath to utilize as the directory in which the output file is created upon saving. Based on the
        default_output_root init argument. If None is given, then defaults to a new dirpath called
        'clair_torch_output' in the directory of the input file.
    cpu_transforms: Transform | Sequence[Transform] | None
        Optional collection of Transforms that will be performed on the data right after reading it from a file.
    """
    @typechecked
    def __init__(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                 default_output_root: Optional[str | Path] = None,
                 cpu_transforms: Optional[Transform | Sequence[Transform]] = None):
        """
        Initializes the instance with the given paths. Output and default roots are optional and defer to a default
        output root if None is given. output_path overrides any possible default_output_root values. Upon loading data
        the given cpu_transforms are performed on the data sequentially.
        Args:
            input_path: the path at which the file is found.
            output_path: the path to which output a modified file.
            default_output_root: a root directory path to utilize if no output_path is given.
            cpu_transforms: Transform(s) to be performed on the data on the cpu-side upon loading the data.
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path is not None else None
        default_output_root = Path(default_output_root) if default_output_root is not None else None

        validate_input_file_path(input_path, None)
        self.input_path = input_path
        if default_output_root is None:
            self.default_output_root = self.input_path.parent.joinpath("clair_torch_output")
        else:
            self.default_output_root = default_output_root

        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = self.default_output_root.joinpath(self.input_path.name)

        if not is_potentially_valid_file_path(self.output_path):
            raise IOError(f"Invalid path for your OS: {self.output_path}")

        if cpu_transforms is not None and not isinstance(cpu_transforms, (list, tuple)):
            cpu_transforms = [cpu_transforms]

        if cpu_transforms is not None and not all(isinstance(t, Transform) for t in cpu_transforms):
            raise TypeError(f"At least one item in transforms is of incorrect type: {cpu_transforms}")

        self.cpu_transforms = cpu_transforms

    @abstractmethod
    def get_input_paths(self) -> Path | tuple[Path, ...]:
        """
        Method for getting the input path(s) from a FileSettings class.
        Returns:
            A single Path or tuple of Paths.
        """
        pass

    @abstractmethod
    def get_output_paths(self) -> Path | tuple[Path, ...]:
        """
        Method for getting the output path(s) from a FileSettings class.
        Returns:
            A single Path or tuple of Paths.
        """
        pass

    @abstractmethod
    def get_transforms(self) -> List[Transform] | None:
        """
        Method for getting the possible Transform operations from a FileSettings class.
        Returns:
            A list of Transforms or None if no Transforms are given.
        """
        pass

    @abstractmethod
    def get_numeric_metadata(self) -> dict:
        """
        Method for getting numeric metadata associated with a file. Should always return at least an empty dict.
        Returns:
            dict[str, int | float]
        """
        pass

    @abstractmethod
    def get_text_metadata(self) -> dict:
        """
        Method for getting the text metadata associated with a file. Should always return at least an empty dict.
        Returns:
            dict[str, str]
        """
        pass

    @abstractmethod
    def get_all_metadata(self) -> dict:
        """
        Method for getting all the metadata associated with a file. Should always return at least an empty dict.
        Returns:
            dict[str, int | float | str]
        """
        pass
