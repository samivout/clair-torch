"""
Module for the image related dataset classes that inherit torch.Dataset through the subpackage's base classes.
"""
from typing import Sequence, List

import torch
from typeguard import typechecked

from clair_torch.common.file_settings import FileSettings, FrameSettings, PairedFrameSettings
from clair_torch.datasets.base import MultiFileMapDataset, MultiFileArtefactMapDataset
from clair_torch.datasets.collate import custom_collate
from clair_torch.common.enums import MissingStdMode, DarkFieldMode, MissingValMode


class ImageMapDataset(MultiFileMapDataset):
    @typechecked
    def __init__(self, files: tuple[FrameSettings | PairedFrameSettings, ...], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT,
                 missing_std_value: float = 0.0, default_get_item_key: str = "raw",
                 missing_val_mode: MissingValMode = MissingValMode.ERROR):
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
        super().__init__(files=files, copy_preloaded_data=copy_preloaded_data, missing_std_mode=missing_std_mode,
                         missing_std_value=missing_std_value, default_getitem_key=default_get_item_key,
                         missing_val_mode=missing_val_mode)


class FlatFieldArtefactMapDataset(MultiFileArtefactMapDataset):
    @typechecked
    def __init__(self, files: tuple[FrameSettings | PairedFrameSettings, ...], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT, missing_std_value: float = 0.0,
                 attributes_to_match: dict[str, None | int | float] = None,
                 cache_size: int = 0, missing_val_mode: MissingValMode = MissingValMode.ERROR,
                 default_get_item_key: str = "raw"):
        """
        Dataset class for handling calibration images. Currently, mainly used for flat-field correction.
        Args:
            attributes_to_match:
            copy_preloaded_data:
            files: list of FrameSettings objects composing the dataset of calibration images.
            missing_std_mode: how missing uncertainty images should be dealt with. Read more in .enums.MissingStdMode.
            missing_std_value: a constant that is used in a manner defined by the missing_std_mode to deal with missing
                uncertainty images.
        """
        # Workaround for mutable default argument.
        if attributes_to_match is None:
            attributes_to_match = {"magnification": None, "illumination": None}

        super().__init__(files=files, copy_preloaded_data=copy_preloaded_data, missing_std_mode=missing_std_mode,
                         missing_std_value=missing_std_value, attributes_to_match=attributes_to_match,
                         cache_size=cache_size, missing_val_mode=missing_val_mode,
                         default_get_item_key=default_get_item_key)

    def _get_matching_image_settings_idx(self, reference_image_settings: FrameSettings,
                                         matching_attributes: dict[str, int | float | None]) -> int | None:
        """
        Internal helper method for getting a matching FrameSettings object. If no matches are found, returns None.
        Args:
            reference_image_settings: the FrameSettings object for which to find a match.
            matching_attributes: the attributes that should match between the reference and one of the FrameSettings
                contained.

        Returns:
            If a match is found, returns the index of that FrameSettings object. If no matches are found, returns None.
        """
        for i, image_settings in enumerate(self.files):

            if image_settings.is_match(reference_image_settings, attributes=matching_attributes):
                return i

        return None


class DarkFieldArtefactMapDataset(MultiFileArtefactMapDataset):
    @typechecked
    def __init__(self, files: tuple[FrameSettings | PairedFrameSettings, ...], copy_preloaded_data: bool = True,
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT, missing_std_value: float = 0.0,
                 attributes_to_match: dict[str, None | int | float] = None, cache_size: int = 0,
                 missing_val_mode: MissingValMode = MissingValMode.ERROR,
                 dark_field_mode: DarkFieldMode = DarkFieldMode.MAX, default_get_item_key: str = "raw"):
        """
        Dataset class for handling calibration images. Currently, mainly used for flat-field correction.
        Args:
            files: list of FrameSettings objects composing the dataset of calibration images.
            missing_std_mode: how missing uncertainty images should be dealt with. Read more in .enums.MissingStdMode.
            missing_std_value: a constant that is used in a manner defined by the missing_std_mode to deal with missing
                uncertainty images.
        """

        # Workaround for mutable default argument.
        if attributes_to_match is None:
            attributes_to_match = {"exposure_time": 0.05}

        super().__init__(files=files, copy_preloaded_data=copy_preloaded_data, missing_std_mode=missing_std_mode,
                         missing_std_value=missing_std_value, attributes_to_match=attributes_to_match,
                         cache_size=cache_size, missing_val_mode=missing_val_mode,
                         default_get_item_key=default_get_item_key)

        self.dark_field_mode = dark_field_mode

    def _get_matching_image_settings_idx(self, reference_image_settings: FrameSettings,
                                         matching_attributes: dict[str, int | float | None]) -> tuple[int, str] | None:
        """
        Internal helper method for getting a matching FrameSettings object. If no matches are found, returns None.
        Args:
            reference_image_settings: the FrameSettings object for which to find a match.
            matching_attributes: the attributes that should match between the reference and one of the FrameSettings
                contained.

        Returns:
            If a match is found, returns the index of that FrameSettings object. If no matches are found, returns None.
        """
        # If max mode, return the index of the last image, which is the one with the largest exposure time.
        if self.dark_field_mode == DarkFieldMode.MAX:
            return len(self) - 1, "exposure_time"

        # Else check for match.
        for i in self.sorted_indices["exposure_time"]:

            image_settings = self.files[i]
            if image_settings.is_match(reference_image_settings, attributes=matching_attributes):
                return i, "exposure_time"

        return None


if __name__ == "__main__":
    pass
