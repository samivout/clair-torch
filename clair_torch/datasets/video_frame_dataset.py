"""
Module for the video related dataset class, which inherits the IterableDataset from torch.utils.data.
"""

from typeguard import typechecked

from clair_torch.common.file_settings import FrameSettings, PairedFrameSettings
from clair_torch.common.enums import MissingStdMode
from clair_torch.common.data_io import load_video_frames_generator

import torch


class VideoIterableDataset(torch.utils.data.IterableDataset):
    @typechecked
    def __init__(self, frame_settings: tuple[FrameSettings | PairedFrameSettings, ...],
                 missing_std_mode: MissingStdMode = MissingStdMode.CONSTANT, missing_std_value: float = 0.0):
        """
        # TODO: add handling for std files along the main value files.
        Dataset class for video files. Treats all encompassed files as a single dataset, jumping smoothly from one file
        to the next upon exhausting the frames from one file.
        Args:
            frame_settings: list of FrameSettings objects composing the dataset.
            missing_std_mode: enum flag determining how missing uncertainty images should be handled.
            missing_std_value: a constant that is used in a manner defined by the missing_std_mode to deal with missing
                uncertainty images.
        """
        self.frame_settings = frame_settings
        self.missing_std_mode = missing_std_mode
        self.shared_std_tensor = torch.tensor(missing_std_value)
        self._running_index = 0

    def __len__(self):

        number_of_frames = 0
        for frame_setting in self.frame_settings:
            number_of_frames += frame_setting.get_numeric_metadata()["number_of_frames"]
        return number_of_frames

    def __iter__(self) -> tuple[int, torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
        """
        Access method for the frames of this dataset. Iterates through the files and frames, moving on to the next file
        upon exhausting a file.
        Returns:

        """
        for settings in self.frame_settings:

            input_paths = settings.get_input_paths()
            transforms = settings.get_transforms()

            if isinstance(input_paths, tuple):
                val_path, std_path = input_paths
            else:
                val_path = input_paths
                std_path = None

            if isinstance(transforms, tuple):
                val_transforms, std_transforms = transforms
            else:
                val_transforms = transforms
                std_transforms = None

            metadata = settings.get_numeric_metadata()

            try:
                for frame in load_video_frames_generator(val_path, val_transforms):
                    if self.missing_std_mode == MissingStdMode.NONE:
                        std_image = None
                    elif self.missing_std_mode == MissingStdMode.CONSTANT:
                        std_image = self.shared_std_tensor.expand_as(frame)
                    elif self.missing_std_mode == MissingStdMode.MULTIPLIER:
                        std_image = frame * self.shared_std_tensor
                    else:
                        raise ValueError(f"Unsupported MissingStdMode: {self.missing_std_mode}")

                    self._running_index += 1
                    yield self._running_index - 1, frame, std_image, metadata
            except StopIteration:
                return
        return
