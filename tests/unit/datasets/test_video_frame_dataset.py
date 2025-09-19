import pytest
from unittest.mock import patch, MagicMock

import torch
from typeguard import suppress_type_checks, TypeCheckError

from clair_torch.common.file_settings import FrameSettings
import clair_torch.datasets.video_frame_dataset
from clair_torch.common.enums import MissingStdMode
from clair_torch.datasets.video_frame_dataset import VideoIterableDataset


class TestVideoIterableDataset:

    def test_video_iterable_dataset_iter(self, tmp_path, fake_frame_settings_factory):

        mock_transform = MagicMock()
        mock_frame_settings = []

        val_image = torch.ones((3, 3, 3), dtype=torch.float64)

        input_paths = [tmp_path / "file1.avi", tmp_path / "file2.avi", tmp_path / "file3.avi"]
        transforms = [mock_transform, mock_transform, mock_transform]
        numeric_metadata = [{"exposure_time": 1.0, "number_of_frames": 2},
                            {"exposure_time": 3.0, "number_of_frames": 2},
                            {"exposure_time": 0.5, "number_of_frames": 2}]

        for input_path, transform, numeric_meta in zip(input_paths, transforms, numeric_metadata):
            fake_frame_settings = fake_frame_settings_factory(numeric_meta, input_path, None, transform)
            mock_frame_settings.append(fake_frame_settings)

        mock_frame_settings = tuple(mock_frame_settings)

        with suppress_type_checks():
            video_dataset = VideoIterableDataset(
                frame_settings=mock_frame_settings, missing_std_mode=MissingStdMode.CONSTANT,
                missing_std_value=2.0
            )

        # Lambda is used to mock the load_video_frames_generator function call.
        load_video_patch = patch.object(clair_torch.datasets.video_frame_dataset,
                                        clair_torch.datasets.video_frame_dataset.load_video_frames_generator.__name__,
                                        side_effect=lambda *a, **k: (x for x in (val_image, val_image)))

        returned_batches = []

        with load_video_patch as mock_generator:

            for batch in video_dataset:
                returned_batches.append(batch)

        assert mock_generator.call_count == 3
        assert len(returned_batches) == 6
        assert len(video_dataset) == 6

        assert numeric_metadata[0]["exposure_time"] == returned_batches[0][3]["exposure_time"]
        assert numeric_metadata[0]["exposure_time"] == returned_batches[1][3]["exposure_time"]
        assert numeric_metadata[1]["exposure_time"] == returned_batches[2][3]["exposure_time"]
        assert numeric_metadata[1]["exposure_time"] == returned_batches[3][3]["exposure_time"]
        assert numeric_metadata[2]["exposure_time"] == returned_batches[4][3]["exposure_time"]
        assert numeric_metadata[2]["exposure_time"] == returned_batches[5][3]["exposure_time"]

    def test_video_iterable_dataset_invalid_args(self, tmp_path):

        validate_input_file_patch = patch.object(clair_torch.common.base,
                                                 clair_torch.common.base.validate_input_file_path.__name__,
                                                 return_value=True)

        with validate_input_file_patch:
            frame_settings = (FrameSettings(input_path=tmp_path / "file_1"),)

        bad_frame_settings = "this_is_bad"
        bad_missing_std_mode = "this_is_bad"
        bad_missing_std_value = "this_is_bad"

        missing_std_mode = MissingStdMode.MULTIPLIER
        missing_std_value = 1.0

        with pytest.raises(TypeCheckError):
            _ = VideoIterableDataset(bad_frame_settings, missing_std_mode, missing_std_value)

        with pytest.raises(TypeCheckError):
            _ = VideoIterableDataset(frame_settings, bad_missing_std_mode, missing_std_value)

        with pytest.raises(TypeCheckError):
            _ = VideoIterableDataset(frame_settings, missing_std_mode, bad_missing_std_value)
