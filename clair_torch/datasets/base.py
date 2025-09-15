"""
Module for the base classes of the datasets subpackage.
"""
from typing import TypeVar, List
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, IterableDataset

from clair_torch.validation.type_checks import validate_all
from clair_torch.common.enums import MissingStdMode, MissingValMode
from clair_torch.common.mixins import Clearable
from clair_torch.common.file_settings import FrameSettings, PairedFrameSettings
from clair_torch.common.data_io import load_image
from clair_torch.datasets.collate import custom_collate


T_File = TypeVar("T_File", bound=FrameSettings)


class MultiFileMapDataset(Dataset[T_File], ABC):
    """
    A generic base class for map-style Dataset classes. Dataset classes must manage files via a concrete implementation
    of the generic base FileSettings class.
    """
    def __init__(self, files: tuple[T_File], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT, missing_std_value: float = 0.0,
                 default_getitem_key="raw", missing_val_mode: MissingValMode = MissingValMode.ERROR):

        validate_all(files, (FrameSettings, PairedFrameSettings), allow_none_iterable=False, allow_none_elements=False,
                     raise_error=True)
        validate_all((copy_preloaded_data,), bool, allow_none_iterable=False, allow_none_elements=False,
                     raise_error=True)
        validate_all((missing_std_value,), (int, float), allow_none_iterable=False, allow_none_elements=False,
                     raise_error=True)
        validate_all((default_getitem_key,), str, allow_none_elements=False, allow_none_iterable=False,
                     raise_error=True)

        self.files = files
        self.preloaded_dataset = None
        self.copy_preloaded_data = copy_preloaded_data
        self.missing_std_mode = missing_std_mode
        self.missing_std_value = missing_std_value
        self.shared_std_tensor = torch.tensor(missing_std_value)
        self.missing_val_mode = missing_val_mode

        # Create a dictionary of numeric metadata keys, based on which to create lists of indices of the files in
        # self.file based on the sorting of each key.
        self.sorted_indices = {
            key: sorted(range(len(self.files)), key=lambda i: self.files[i].get_numeric_metadata()[key]) for key in files[0].get_numeric_metadata().keys()
        }

        # Insert a raw sorting order in the indices.
        self.sorted_indices["raw"] = list(range(len(self.files)))

        if default_getitem_key not in self.sorted_indices.keys():
            raise ValueError(f"The default access key {default_getitem_key} is not found in the dataset.")
        self.default_getitem_key = default_getitem_key

    def __len__(self) -> int:
        """
        The length of the dataset is defined as the number of files in it manages.
        Returns:
            int representing the number of files.
        """
        return len(self.files)

    def __getitem__(self, key) -> tuple[int, torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
        """
        This method loads images from disk with OpenCV, converts them to PyTorch tensors, runs them through the given
        transformations, finally returning the image tensor and a scalar tensor of the exposure time. It also falls
        back on the preloaded tensors if they are available. This method should be used as the main way to access the
        tensors.
        Args:
            key: index of the item to get.

        Returns:
            A tuple (tensor, tensor | None, dict[str, float | int]), representing the value image, optional uncertainty
            image and a numeric metadata dictionary.
        """
        # Support (idx, meta_key) indexing
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Expected (index, meta_key) for tuple indexing.")
            idx, meta_key = key
        else:
            idx, meta_key = key, self.default_getitem_key

        # Resolve sorted index
        sorted_idx = self.sorted_indices[meta_key][idx]

        # Utilize the preloaded dataset if it exists.
        if self.preloaded_dataset is not None:
            indices, val_tensors, std_tensors, numeric_metadatas = self.preloaded_dataset
            if self.copy_preloaded_data:
                return indices[sorted_idx], val_tensors[sorted_idx].clone(), std_tensors[sorted_idx].clone(), deepcopy(numeric_metadatas[sorted_idx])
            else:
                return indices[sorted_idx], val_tensors[sorted_idx], std_tensors[sorted_idx], numeric_metadatas[sorted_idx]

        # Load images normally
        val_image, std_image, numeric_metadatas = self._load_value_and_std_image(sorted_idx)

        return sorted_idx, val_image, std_image, numeric_metadatas

    def _load_value_and_std_image(self, idx: int) \
            -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
        """
        Shared image loading function that handles the loading of both the value images and uncertainty images and the
        numeric metadata.
        Args:
            idx: index of the managed file to load images off of.

        Returns:
            Tuple of (tensor, tensor | None, dict) representing the value image, possible uncertainty image and numeric
            metadata dictionary.
        """
        file_settings = self.files[idx]
        input_paths = file_settings.get_input_paths()
        transforms = file_settings.get_transforms()

        if isinstance(transforms, tuple):
            val_transforms, std_transforms = transforms
        else:
            val_transforms = transforms
            std_transforms = None

        if isinstance(input_paths, tuple):
            val_path, std_path = input_paths
        else:
            val_path = input_paths
            std_path = None

        val_image = load_image(val_path, val_transforms)
        if std_path is not None:
            std_image = load_image(std_path, std_transforms)
        else:
            if self.missing_std_mode == MissingStdMode.NONE:
                std_image = None
            elif self.missing_std_mode == MissingStdMode.CONSTANT:
                std_image = self.shared_std_tensor.expand_as(val_image)
            elif self.missing_std_mode == MissingStdMode.MULTIPLIER:
                std_image = val_image * self.shared_std_tensor
            else:
                raise ValueError(f"Unsupported MissingStdMode: {self.missing_std_mode}")

        numeric_metadata = file_settings.get_numeric_metadata()

        return val_image, std_image, numeric_metadata

    def preload_dataset(self) -> None:
        """
        Loads all data from disk into memory and stores them as a tuple of lists of tensors in self._preloaded_dataset.
        This method utilizes the __getitem__ method.
        """
        indices = []
        val_tensors = []
        std_tensors = []
        numeric_metadata = []

        for i in range(len(self)):
            index, val_tensor, std_tensor, numeric_metadata_tensor = self[i]
            indices.append(index)
            val_tensors.append(val_tensor)
            std_tensors.append(std_tensor)
            numeric_metadata.append(numeric_metadata_tensor)

        self.preloaded_dataset = (indices, val_tensors, std_tensors, numeric_metadata)

    def _get_closest_frame_settings_idx(self, reference_frame_settings: FrameSettings, attribute: str) -> int | None:
        """
        Method for getting the closest possible match from the dataset for a given reference FrameSettings instance as
        determined by the given attribute.
        Args:
            reference_frame_settings: the FrameSettings instance for which to find the closest match.
            attribute: the attribute based on which the match is searched for.

        Returns:
            The index of the closest matching FrameSettings instance in this dataset, or None if dataset is empty.
        """

        ...


class MultiFileArtefactMapDataset(MultiFileMapDataset[T_File], Clearable, ABC):

    _CLEARABLE_ATTRIBUTES = {"_cache": OrderedDict}

    def __init__(self, files: tuple[T_File], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT,
                 missing_std_value: float = 0.0, attributes_to_match: dict[str, None | int | float] = None,
                 cache_size: int = 0, missing_val_mode: MissingValMode = MissingValMode.ERROR,
                 default_get_item_key: str = "raw"):

        MultiFileMapDataset.__init__(self, files=files, copy_preloaded_data=copy_preloaded_data,
                                     missing_std_mode=missing_std_mode, missing_std_value=missing_std_value,
                                     missing_val_mode=missing_val_mode,
                                     default_getitem_key=default_get_item_key)
        Clearable.__init__(self, MultiFileArtefactMapDataset._CLEARABLE_ATTRIBUTES)

        self.attributes_to_match = attributes_to_match
        self.cache_size = cache_size
        self._cache: OrderedDict[int, tuple[int, torch.Tensor, torch.Tensor | None, dict]] = OrderedDict()

    def __getitem__(self, key):
        # Support (idx, meta_key) indexing
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Expected (index, meta_key) for tuple indexing.")
            idx, meta_key = key
        else:
            idx, meta_key = key, self.default_getitem_key

        sorted_idx = self.sorted_indices[meta_key][idx]
        cache_key = (sorted_idx, meta_key)

        if cache_key in self._cache:
            # Move accessed item to end (most recently used)
            index, val, std, meta = self._cache.pop(cache_key)
            self._cache[cache_key] = (index, val, std, meta)
            return index, val, std, meta

        # Fall back to normal loading
        index, val, std, meta = super().__getitem__((idx, meta_key))

        # Insert into cache
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)  # Evict least recently used
            self._cache[cache_key] = (index, val, std, meta)

        return index, val, std, meta

    def get_matching_artefact_images(self, reference_frame_settings_list: list[FrameSettings]) \
            -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict | None]:
        """
        Find matching artefact images for multiple reference FrameSettings objects and collate the results into batches.
        Args:
            reference_frame_settings_list: list of FrameSettings objects.

        Returns:
            A tuple (indices, val_batch, std_batch, meta_batch), where:
                - indices: batched indices of matching images (tensor).
                - val_batch: batched value images (tensor).
                - std_batch: batched std images (tensor or None).
                - meta_batch: batched metadata dict.
        """
        validate_all(reference_frame_settings_list, (FrameSettings, PairedFrameSettings), raise_error=True,
                     allow_none_iterable=False, allow_none_elements=False)

        results = []
        for ref in reference_frame_settings_list:

            result = self._get_matching_artefact_image(ref)

            if result is None and self.missing_val_mode == MissingValMode.SKIP_BATCH:
                print("Skipped batch.")
                return None, None, None, None

            if result is None and self.missing_val_mode == MissingValMode.ERROR:
                raise RuntimeError("No matching artefact images found.")

            results.append(result)

        return custom_collate(results)

    def _get_matching_artefact_image(self, reference_frame_settings: FrameSettings) \
            -> None | tuple[int, torch.Tensor, torch.Tensor | None, dict[str, float | int] | None]:
        """
        Get a matching calibration image for the given reference_image_settings object, based on the given
        matching_attributes. Returns the first matching FrameSettings object from the contained FrameSettings objects.
        The returned images are unsqueezed from shape (C, H, W) to (N, C, H, W), where N=1 to match the shape of images
        returned from DataLoaders.
        Args:
            reference_frame_settings: the FrameSettings object for which to find a match.

        Returns:
            A tuple (val_image, std_image, numeric_metadata), representing the calibration value image, uncertainty image
            and a dictionary of the associated numeric metadata. Each of these can be None if no matches are found, or
            only std_image can be None if it is not found and MissingStdMode.NONE is used.
        """
        ret = self._get_matching_image_settings_idx(reference_frame_settings, self.attributes_to_match)

        if ret is not None:

            index, val_image, std_image, numeric_metadata = self[ret]

            return index, val_image, std_image, numeric_metadata

        return None

    @abstractmethod
    def _get_matching_image_settings_idx(self, reference_image_settings: FrameSettings,
                                         matching_attributes: dict[str, int | float | None]) -> tuple[int, str] | None:
        """
        Internal helper method for getting a matching FrameSettings object. If no matches are found, returns None. The
        inheriting class should implement the logic how a match is made.
        Args:
            reference_image_settings: the FrameSettings object for which to find a match.
            matching_attributes: the attributes that should match between the reference and one of the FrameSettings
                contained.

        Returns:
            If a match is found, returns the index of that FrameSettings object. If no matches are found, returns None.
        """
        ...


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
