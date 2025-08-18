"""
Module for the image related dataset classes that inherit torch.Dataset through the subpackage's base classes.
"""
from typing import Sequence, List
from copy import deepcopy

import torch

from clair_torch.common.file_settings import FileSettings, FrameSettings
from clair_torch.common.data_io import load_image
from clair_torch.datasets.base import MultiFileMapDataset
from clair_torch.common.enums import MissingStdMode


class ImageMapDataset(MultiFileMapDataset[FileSettings]):

    def __init__(self, files: List[FileSettings], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT,
                 missing_std_value: float = 0.0):
        """
        ImageDataset is the master image data object. The files attribute holds a list of FileSettings-based objects.
        The image tensors shapes are (C, H, W), that is number of channels, height and width. Through a DataLoader the
        shape is expanded into (N, C, H, W) with N standing for the number of images in the batch.
        Args:
            files: list of the FileSettings-based objects composing the dataset.
            copy_preloaded_data: whether preloaded data should be returned as a new copy or as a reference to the
                preloaded data contained in self._preloaded_dataset.
            missing_std_mode: how missing uncertainty images should be dealt with. Read more in .enums.MissingStdMode.
            missing_std_value: a constant that is used in a manner defined by the missing_std_mode to deal with missing
                uncertainty images.
        """
        super().__init__(files)

        self._sort_image_settings_by_exposure_time()
        self._preloaded_dataset = None
        self.copy_preloaded_data = copy_preloaded_data
        self.missing_std_mode = missing_std_mode
        self.shared_std_tensor = torch.tensor(missing_std_value)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
        """
        This method loads images from disk with OpenCV, converts them to PyTorch tensors, runs them through the given
        transformations, finally returning the image tensor and a scalar tensor of the exposure time. It also falls
        back on the preloaded tensors if they are available.
        Args:
            idx: index of the item to get.

        Returns:

        """
        if self._preloaded_dataset is not None:
            val_tensor, std_tensor, numeric_metadata = self._preloaded_dataset
            if self.copy_preloaded_data:
                return val_tensor[idx].clone(), std_tensor[idx].clone(), deepcopy(numeric_metadata[idx])
            else:
                return val_tensor[idx], std_tensor[idx], numeric_metadata[idx]

        val_image, std_image, numeric_metadata = _load_value_and_std_image(self.files[idx], self.missing_std_mode,
                                                                           self.shared_std_tensor)

        return val_image, std_image, numeric_metadata

    def preload_dataset(self) -> None:
        """
        Loads all data from disk into memory and stores them as a tuple of lists of tensors in self._preloaded_dataset.
        This method utilizes the __getitem__ method.
        """
        val_tensors = []
        std_tensors = []
        numeric_metadata = []

        for i in range(len(self)):
            val_tensor, std_tensor, numeric_metadata_tensor = self[i]
            val_tensors.append(val_tensor)
            std_tensors.append(std_tensor)
            numeric_metadata.append(numeric_metadata_tensor)

        self._preloaded_dataset = (val_tensors, std_tensors, numeric_metadata)

    def _sort_image_settings_by_exposure_time(self) -> None:
        """
        Utility method for ensuring sorting of the files in increasing exposure time order, as most operations depend
        on this ordering.
        """
        self.files.sort(key=lambda frame_settings: frame_settings.get_numeric_metadata()['exposure_time'])


class ArtefactMapDataset(MultiFileMapDataset[FrameSettings]):

    def __init__(self, files: List[FrameSettings], missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT,
                 missing_std_value: float = 0.0):
        """
        Dataset class for handling calibration images. Currently, mainly used for flat-field correction.
        Args:
            files: list of FrameSettings objects composing the dataset of calibration images.
            missing_std_mode: how missing uncertainty images should be dealt with. Read more in .enums.MissingStdMode.
            missing_std_value: a constant that is used in a manner defined by the missing_std_mode to deal with missing
                uncertainty images.
        """
        super().__init__(files)

        self.missing_std_mode = missing_std_mode
        self.shared_std_tensor = torch.tensor(missing_std_value)
    
    def __len__(self):
        return len(self.files)

    def get_matching_artefact_image(self, reference_image_settings: FrameSettings,
                                    matching_attributes: Sequence[str])\
            -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float | int] | None]:
        """
        Get a matching calibration image for the given reference_image_settings object, based on the given
        matching_attributes. Returns the first matching FrameSettings object from the contained FrameSettings objects.
        The returned images are unsqueezed from shape (C, H, W) to (N, C, H, W), where N=1 to match the shape of images
        returned from DataLoaders.
        Args:
            reference_image_settings: the FrameSettings object for which to find a match.
            matching_attributes: the attributes that should match between the reference and one of the FrameSettings
                contained.

        Returns:
            A tuple (val_image, std_image, numeric_metadata), representing the calibration value image, uncertainty image
            and a dictionary of the associated numeric metadata. Each of these can be None if no matches are found, or
            only std_image can be None if it is not found and MissingStdMode.NONE is used.
        """
        matching_flatfield_image_settings = self._get_matching_image_settings(reference_image_settings,
                                                                              matching_attributes)

        if matching_flatfield_image_settings is not None:

            val_image, std_image, numeric_metadata = _load_value_and_std_image(matching_flatfield_image_settings,
                                                                               self.missing_std_mode,
                                                                               self.shared_std_tensor)

            # Add batch dimension to artefact image to match the (N, C, H, W) shape of dataloader outputs.
            val_image = val_image.unsqueeze_(0)
            std_image = std_image.unsqueeze_(0) if std_image is not None else None

            return val_image, std_image, numeric_metadata

        return None, None, None

    def _get_matching_image_settings(self, reference_image_settings: FrameSettings,
                                     matching_attributes: Sequence[str]) -> FrameSettings | None:
        """
        Internal helper method for getting a matching FrameSettings object. If no matches are found, returns None.
        Args:
            reference_image_settings: the FrameSettings object for which to find a match.
            matching_attributes: the attributes that should match between the reference and one of the FrameSettings
                contained.

        Returns:
            If a match is found, returns that FrameSettings object, or if no matches are found, returns None.
        """
        for image_settings in self.files:

            if image_settings.is_match(reference_image_settings, attributes=matching_attributes):
                return image_settings

        return None


def _load_value_and_std_image(file_settings: FileSettings, missing_std_mode: MissingStdMode,
                              shared_std_tensor: torch.Tensor)\
        -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
    """
    Shared image loading function that handles the loading of both the value images and uncertainty images and the
    numeric metadata.
    Args:
        file_settings: the FileSettings instance that controls the paths and transforms of the loading.
        missing_std_mode: enum flag determining how missing uncertainty images should be handled.
        shared_std_tensor: the shared std value to use in case of missing uncertainty images.

    Returns:
        Tuple of (tensor, tensor | None, dict) representing the value image, possible uncertainty image and numeric
        metadata dictionary.
    """
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
        if missing_std_mode == MissingStdMode.NONE:
            std_image = None
        elif missing_std_mode == MissingStdMode.CONSTANT:
            std_image = shared_std_tensor.expand_as(val_image)
        elif missing_std_mode == MissingStdMode.MULTIPLIER:
            std_image = val_image * shared_std_tensor
        else:
            raise ValueError(f"Unsupported MissingStdMode: {missing_std_mode}")

    numeric_metadata = file_settings.get_numeric_metadata()

    return val_image, std_image, numeric_metadata


if __name__ == "__main__":
    pass
