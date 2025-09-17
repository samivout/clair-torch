"""
Module for concrete classes used to manage single input-output files and pairs of input-output files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Sequence, Type, Callable

from typeguard import typechecked

from clair_torch.metadata.imaging_metadata import ImagingMetadata, BaseMetadata
from clair_torch.common.transforms import Transform
from clair_torch.common.base import BaseFileSettings
from clair_torch.common.data_io import _get_file_input_paths_by_pattern, _pair_main_and_std_files


class FileSettings(BaseFileSettings):
    """
    Class for managing input and output paths related to an arbitrary file. Main use is to manage the IO settings for
    use inside a PyTorch Dataset class.

    Attributes:
    -----------
    Inherits attributes from BaseFileSettings.
    """

    @typechecked
    def __init__(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                 default_output_root: Optional[str | Path] = None,
                 cpu_transforms: Optional[Transform | Iterable[Transform]] = None):
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
        super().__init__(input_path, output_path, default_output_root, cpu_transforms)

    def get_input_paths(self) -> Path:
        """
        Method for getting the input path.
        Returns:
            Path.
        """
        return self.input_path

    def get_output_paths(self) -> Path:
        """
        Method for getting the output path.
        Returns:
            Path.
        """
        return self.output_path

    def get_candidate_std_output_path(self) -> Path:
        """
        Method for getting a candidate output path for an uncertainty file.
        Returns:
            Constructed uncertainty file output path based on the main input file and its filetype suffix.
        """
        candidate_name = self.input_path.stem + " STD" + self.input_path.suffix
        std_output_path = self.output_path.parent / candidate_name

        return std_output_path

    def get_transforms(self) -> List[Transform] | None:
        """
        Method for getting the possible Transform operations to be performed on the data upon reading it.
        Returns:
            List[Transform] or None, if no Transform operations were given on init.
        """
        return self.cpu_transforms

    def get_numeric_metadata(self) -> dict:
        """
        Unused method stub inherited from the base class. This is used in classes further down the subclass tree.
        Returns:
            Empty dict.
        """
        return {}

    def get_text_metadata(self) -> dict:
        """
        Unused method stub inherited from the base class. This is used in classes further down the subclass tree.
        Returns:
            Empty dict.
        """
        return {}

    def get_all_metadata(self) -> dict:
        """
        Unused method stub inherited from the base class. This is used in classes further down the subclass tree.
        Returns:
            Empty dict.
        """
        return {}


class FrameSettings(FileSettings):
    """
    Class for managing input and output paths related to an arbitrary file, with the addition of managing image related
    metadata. Main use is to manage the IO settings for use inside a PyTorch Dataset class.

    Attributes:
    -----------
    Inherits attributes from FileSettings.
    metadata: BaseMetadata
        an encapsulated instance of a BaseMetadata subclass for managing the metadata related to images.
    """
    @typechecked
    def __init__(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                 default_output_root: Optional[str | Path] = None,
                 cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
                 metadata_cls: Type[BaseMetadata] = ImagingMetadata,
                 *metadata_args, **metadata_kwargs):
        super().__init__(input_path, output_path, default_output_root, cpu_transforms)

        self.metadata = metadata_cls(input_path, *metadata_args, **metadata_kwargs)

    def get_numeric_metadata(self) -> dict[str, float | int | None]:
        """
        Method for getting the numeric metadata managed by the encapsulated Metadata class.
        Returns:
            dict[str, int | float].
        """
        return self.metadata.get_numeric_metadata()

    def get_text_metadata(self) -> dict[str, str | None]:
        """
        Method for getting the text metadata managed by the encapsulated Metadata class.
        Returns:
            dict[str, str].
        """
        return self.metadata.get_text_metadata()

    def get_all_metadata(self) -> dict[str, str | int | float | None]:
        """
        Method for getting all the metadata managed by the encapsulated Metadata class.
        Returns:
            dict[str, str | int | float | None].
        """
        return self.metadata.get_all_metadata()

    @typechecked
    def is_match(self, reference: FrameSettings | PairedFrameSettings | BaseMetadata,
                 attributes: dict[str, Optional[int | float]]) -> bool:
        """
        Method for evaluating whether the metadata contained in a given FramSettings instance or Metadata instance are a
        match based on the given sequence of attributes, which act as keys to the metadata dictionary in a Metadata
        instance.
        Args:
            reference: a FrameSettings instance or a BaseMetadata subclass instance.
            attributes: a sequence of string, which define the dictionary keys, whose associated values must be equal
                for a successful match.

        Returns:
            bool, True for a successful match, False for failed.
        """
        if isinstance(reference, (FrameSettings, PairedFrameSettings)):
            reference_metadata = reference.metadata
        elif isinstance(reference, BaseMetadata):
            reference_metadata = reference
        else:
            return False

        return self.metadata.is_match(reference_metadata, attributes)


class PairedFileSettings(FileSettings):
    """
    Class for managing input and output paths of a pair of arbitrary files. Main use is to handle the paired IO
    operations of a value image and its associated uncertainty image. Composed of two instances of FileSettings.

    Attributes:
    -----------
    Inherits attributes from FileSettings.
    """
    @typechecked
    def __init__(self, val_input_path: str | Path, std_input_path: Optional[str | Path] = None,
                 val_output_path: Optional[str | Path] = None,
                 std_output_path: Optional[str | Path] = None,
                 default_output_root: Optional[str | Path] = None,
                 val_cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
                 std_cpu_transforms: Optional[Transform | Iterable[Transform]] = None):
        """
        Initializes a PairedFileSettings object with the given paths. The class both inherits from FileSettings and is
        a composition of two instances of FileSettings. Referring to self.val_input_path is the same as referring to
        self.val_settings.input_path, same follows for other attributes. PairedFileSettings should only be used when an
        uncertainty input file does exist, even though the std_input_path parameter is Optional. The optionality is left
        to enable easy implicit seeking of std files. Use regular FileSettings if no uncertainty files are to be used.
        Args:
            val_input_path: input path for the main value file.
            std_input_path: input path for the associated uncertainty file.
            val_output_path: output path for the modified value file.
            std_output_path: output path for the modified uncertainty file.
            default_output_root: a root directory path to utilize if no output_path is given.
            val_cpu_transforms: transform operations for the value file.
            std_cpu_transforms transform operations for the uncertainty file.
        """
        super().__init__(val_input_path, val_output_path, default_output_root, val_cpu_transforms)

        self.val_settings = FileSettings(
            val_input_path,
            output_path=val_output_path,
            default_output_root=default_output_root,
            cpu_transforms=val_cpu_transforms
        )

        # This enables implicit seeking of STD files, when no STD input file is directly given.
        if std_input_path is None:
            candidate_name = self.val_settings.input_path.stem + " STD" + self.val_settings.input_path.suffix
            std_input_path = self.val_settings.input_path.parent / candidate_name

        self.std_settings = FileSettings(
            std_input_path,
            output_path=std_output_path,
            default_output_root=self.val_settings.default_output_root,
            cpu_transforms=std_cpu_transforms
        )

    def get_input_paths(self) -> Tuple[Path, Path]:
        """
        Method for getting the input paths of a PairedFileSettings instance. Overrides the inherited method by deferring
        the process for the two encapsulated instances.
        Returns:
            Tuple of Paths, first for value file, second for uncertainty file.
        """
        return self.val_settings.get_input_paths(), self.std_settings.get_input_paths()

    def get_output_paths(self) -> Tuple[Path, Path]:
        """
        Method for getting the output paths of a PairedFileSettings instance. Overrides the inherited method by deferring
        the process for the two encapsulated instances.
        Returns:
            Tuple of paths, first for value file, second for uncertainty file.
        """
        return self.val_settings.get_output_paths(), self.std_settings.get_output_paths()

    def get_transforms(self) -> Tuple[List[Transform], List[Transform]]:
        """
        Method for getting the transformation operations. Overrides the inherited method by deferring the process for
        the two encapsulated instances.
        Returns:
            Tuple of Lists of Transforms, first for value file Transforms, second for uncertainty file Transforms.
        """
        return self.val_settings.get_transforms(), self.std_settings.get_transforms()


class PairedFrameSettings(PairedFileSettings):
    """
    Class for managing paired files with their associated metadatas, based on the PairedFileSettings class.

    Attributes:
    -----------
    Inherits attributes from PairedFileSettings.
    val_metadata: BaseMetadata
        an encapsulated instance of a BaseMetadata subclass for managing the metadata related to the value image.
    std_metadata: BaseMetadata
        an encapsulated instance of a BaseMetadata subclass for managing the metadata related to the possible uncertainty
        image. Typically, these are equal to the val_metadata values, so by default this is not parsed. The parse_std_meta
        init argument is used to determine whether the std_metadata is parsed or not.
    """
    @typechecked
    def __init__(self, val_input_path: str | Path, std_input_path: Optional[str | Path] = None,
                 val_output_path: Optional[str | Path] = None, std_output_path: Optional[str | Path] = None,
                 default_output_root: Optional[str | Path] = None,
                 val_cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
                 std_cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
                 parse_std_meta: bool = False, metadata_cls: Type[BaseMetadata] = ImagingMetadata,
                 *metadata_args, **metadata_kwargs):
        """
        File settings related aspects are all delegated to the PairedFileSettings super class. Additional responsibility
        for this class is maintaining and allowing access to the metadatas of each of the encapsulated files via
        subclasses of BaseMetadata.
        Args:
            val_input_path: input path for the main value file.
            std_input_path: input path for the associated uncertainty file.
            val_output_path: output path for the modified value file.
            std_output_path: output path for the modified uncertainty file.
            default_output_root: a root directory path to utilize if no output_path is given.
            val_cpu_transforms: transform operations for the value file.
            std_cpu_transforms transform operations for the uncertainty file.
            parse_std_meta: whether to parse a Metadata class instance for the STD file.
            metadata_cls: a subclass of BaseMetadata
            *metadata_args: additional args to pass to instantiating the given metadata_cls.
            **metadata_kwargs: additional kwargs to pass to instantiating the given metadata_cls.
        """
        super().__init__(val_input_path, std_input_path, val_output_path, std_output_path, default_output_root,
                         val_cpu_transforms, std_cpu_transforms)

        self.val_metadata = metadata_cls(self.val_settings.input_path, *metadata_args, **metadata_kwargs)

        self.std_metadata = (
            metadata_cls(self.std_settings.input_path, *metadata_args, **metadata_kwargs)
            if parse_std_meta and self.std_settings is not None
            else None
        )

    def get_numeric_metadata(self) -> dict[str, float | int | None]:
        """
        Method for getting the numeric metadata of the value image.
        Returns:
            dict[str, float | int | None]
        """
        return self.val_metadata.get_numeric_metadata()

    def get_text_metadata(self) -> dict[str, str | None]:
        """
        Method for getting the text metadata of the value image.
        Returns:
            dict[str, str | None]
        """
        return self.val_metadata.get_text_metadata()

    def get_all_metadata(self) -> dict[str, str | int | float]:
        """
        Method for getting all the metadata of the value image.
        Returns:
            dict[str, str | int | float | None]
        """
        return self.val_metadata.get_all_metadata()

    @typechecked
    def is_match(self, reference: FrameSettings | PairedFrameSettings | BaseMetadata,
                 attributes: dict[str, Optional[int | float]]) -> bool:
        """
        Method for evaluating whether the metadata contained in a given FramSettings instance or Metadata instance are a
        match based on the given sequence of attributes, which act as keys to the metadata dictionary in a Metadata
        instance. Utilizes the val_metadata associated with the value image.
        Args:
            reference: a FrameSettings instance or a BaseMetadata subclass instance.
            attributes: a sequence of string, which define the dictionary keys, whose associated values must be equal
                for a successful match.

        Returns:
            bool, True for a successful match, False for failed.
        """
        if isinstance(reference, (FrameSettings, PairedFrameSettings)):
            reference_meta = reference.val_metadata
        elif isinstance(reference, BaseMetadata):
            reference_meta = reference
        else:
            return False

        return self.val_metadata.is_match(reference_meta, attributes)


@typechecked
def file_settings_constructor(
    dir_paths: Path | Sequence[Path],
    file_pattern: str,
    recursive: bool = False,
    default_output_root: Optional[Path] = None,
    val_cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
    std_cpu_transforms: Optional[Transform | Iterable[Transform]] = None,
    metadata_cls: Optional[Type[BaseMetadata]] = None,
    strict_exclusive: bool = True,
    *metadata_args,
    **metadata_kwargs
):
    """
    Utility function for creating instances of FileSettings, PairedFileSettings, FrameSettings and PairedFrameSettings
    classes. FrameSettings classes are created when a metadata class is provided, otherwise FileSettings are created.
    The created objects are assigned to three categories, each having a list of their own to hold the objects:
    1. Paired files, containing either PairedFrameSettings or PairedFileSettings objects. 2. Main files, containing
    either FrameSettings or FileSettings objects. 3. Uncertainty files, containing either FrameSettings or FileSettings
    objects. The strict_exclusive parameter controls whether a file can be present in the objects of multiple categories
    of only one. In strict mode, for example, the main file and uncertainty file present in a PairedSettings object
    aren't allowed to also be present in a Settings object in the main and uncertainty categories.
    Args:
        dir_paths: one or multiple paths from which to collect files for creating objects.
        file_pattern: a regex pattern for the file search. Use '*.png' for example to search for any .png file.
        recursive: whether to extend the file search recursively to subdirectories of the given paths.
        default_output_root: optional default output root directory for the created objects.
        val_cpu_transforms: optional main file transform operations to attach to the created objects.
        std_cpu_transforms: optional uncertainty file transform operations to attach to the created objects.
        metadata_cls: a subclass of BaseMetadata to use in creating FrameSettings and PairedFrameSettings objects.
        strict_exclusive: whether to allow a particular file to exist in multiple object categories, or only in the
            highest priority one.
        *metadata_args: args for the instantiation of the given metadata_cls.
        **metadata_kwargs: kwargs for the instantiation of the given metadata_cls.

    Returns:

    """
    def make_single_settings(cls: Callable, path_key: int, cpu_transforms, path_pair):
        return cls(
            input_path=path_pair[path_key],
            default_output_root=default_output_root,
            cpu_transforms=cpu_transforms,
            metadata_cls=metadata_cls if cls is FrameSettings else None,
            *metadata_args,
            **metadata_kwargs
        )

    def make_paired_settings(cls: Callable, pair):
        return cls(
            val_input_path=pair[0],
            std_input_path=pair[1],
            default_output_root=default_output_root,
            val_cpu_transforms=val_cpu_transforms,
            std_cpu_transforms=std_cpu_transforms,
            metadata_cls=metadata_cls if cls is PairedFrameSettings else None,
            *metadata_args,
            **metadata_kwargs
        )

    search_paths = [dir_paths] if isinstance(dir_paths, Path) else dir_paths

    all_paired, all_main, all_std = [], [], []

    for dir_path in search_paths:
        current_files = _get_file_input_paths_by_pattern(dir_path, file_pattern, recursive)
        paired, main, std = _pair_main_and_std_files(current_files, strict_exclusive=strict_exclusive)
        all_paired.extend(paired)
        all_main.extend(main)
        all_std.extend(std)

    if metadata_cls:
        paired_cls, frame_cls = PairedFrameSettings, FrameSettings
    else:
        paired_cls, frame_cls = PairedFileSettings, FileSettings

    paired_settings = tuple(make_paired_settings(paired_cls, p) for p in all_paired)
    main_settings = tuple(make_single_settings(frame_cls, 0, val_cpu_transforms, m) for m in all_main)
    std_settings = tuple(make_single_settings(frame_cls, 1, std_cpu_transforms, s) for s in all_std)

    return paired_settings, main_settings, std_settings


@typechecked
def group_frame_settings_by_attributes(list_of_frame_settings: List[FrameSettings],
                                       attributes: dict[str, None | int | float]) \
        -> List[Tuple[dict[str, str | float | int], List[FrameSettings]]]:
    """
    Sort FrameSettings objects into separate groups based on the values of the given attributes.
    Args:
        list_of_frame_settings: list of the FrameSettings to sort.
        attributes: the attributes to base the grouping on.

    Returns:
        List of tuples, the first item in the tuple containing a dictionary of the attributes used to generate that
        group. The second item in the tuple contains a list of the FrameSettings objects belonging to that group.
    """

    list_of_grouped_frame_settings = []
    list_of_group_metas = []

    for frame_settings in list_of_frame_settings:

        current_metas = frame_settings.get_all_metadata()

        # Generate the first group automatically.
        if not list_of_grouped_frame_settings:

            group_list = [frame_settings]
            list_of_grouped_frame_settings.append(group_list)
            list_of_group_metas.append({k: current_metas[k] for k in attributes if k in current_metas})
            continue

        # Loop through the current groups and check if the current frame_settings fits into any of them.
        number_of_groups = len(list_of_grouped_frame_settings)
        for i, group_list in enumerate(list_of_grouped_frame_settings):

            current_target_metadata = group_list[0].metadata
            candidate_metadata = frame_settings.metadata

            # Add to existing group.
            if current_target_metadata.is_match(candidate_metadata, attributes):
                group_list.append(frame_settings)
                break
            # Generate a new group and add to it.
            if number_of_groups - 1 - i == 0:
                additional_group_list = [frame_settings]
                list_of_grouped_frame_settings.append(additional_group_list)
                list_of_group_metas.append({k: current_metas[k] for k in attributes if k in current_metas})
                break

    return list(zip(list_of_group_metas, list_of_grouped_frame_settings))


if __name__ == "__main__":
    pass
