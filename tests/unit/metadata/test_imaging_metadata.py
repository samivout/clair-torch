import pytest
from unittest.mock import patch

from typeguard import TypeCheckError

from clair_torch.metadata.imaging_metadata import ImagingMetadata, VideoMetadata


class TestImagingMetadata:

    def test_imaging_metadata_init_success(self, tmp_path):

        val_input_path = tmp_path / "50ms 10x DF dummy.tif"

        metadata = ImagingMetadata(val_input_path)

        assert metadata.exposure_time == 0.050
        assert metadata.illumination == 'df'
        assert metadata.magnification == 10.0
        assert metadata.subject == "dummy"

    def test_imaging_metadata_get_numeric_metadata(self, tmp_path):

        val_input_path = tmp_path / "50ms 10x DF dummy.tif"

        metadata = ImagingMetadata(val_input_path)

        numeric_meta_dict = metadata.get_numeric_metadata()

        assert numeric_meta_dict["magnification"] == 10.0
        assert numeric_meta_dict["exposure_time"] == 0.05

    def test_imaging_metadata_get_text_metadata(self, tmp_path):

        val_input_path = tmp_path / "50ms 10x DF dummy.tif"

        metadata = ImagingMetadata(val_input_path)

        text_meta_dict = metadata.get_text_metadata()

        assert text_meta_dict["illumination"] == 'df'
        assert text_meta_dict["subject"] == 'dummy'

    def test_imaging_metadata_get_all_metadata(self, tmp_path):
        val_input_path = tmp_path / "50ms 10x DF dummy.tif"

        metadata = ImagingMetadata(val_input_path)

        full_metadata_dict = metadata.get_all_metadata()

        assert full_metadata_dict["illumination"] == 'df'
        assert full_metadata_dict["subject"] == 'dummy'
        assert full_metadata_dict["magnification"] == 10.0
        assert full_metadata_dict["exposure_time"] == 0.05

    def test_imaging_metadata_invalid_input_path(self):

        val_input_path = 1

        with pytest.raises(TypeCheckError):
            _ = ImagingMetadata(val_input_path)


class TestVideoMetadata:

    def test_video_metadata_init_success(self, tmp_path):
        val_input_path = tmp_path / "50ms 10x DF dummy.tif"

        with patch("clair_torch.metadata.imaging_metadata._get_frame_count", return_value=10):
            metadata = VideoMetadata(val_input_path)

        assert metadata.number_of_frames == 10


