import pytest
from unittest.mock import MagicMock, patch

import torch

from clair_torch.datasets import image_dataset as id
from clair_torch.common.enums import MissingStdMode


class TestImageMapDataset:

    def test_image_map_dataset(self, tmp_path, fake_frame_settings_factory):

        mock_transform = MagicMock()
        mock_file_settings = []

        val_image = torch.ones((3, 3), dtype=torch.float64)

        input_paths = [tmp_path / "file1.tif", tmp_path / "file2.tif", tmp_path / "file3.tif"]
        transforms = [mock_transform, mock_transform, mock_transform]
        numeric_metadata = [{"exposure_time": 1.0}, {"exposure_time": 3.0}, {"exposure_time": 0.5}]

        for input_path, transform, numeric_meta in zip(input_paths, transforms, numeric_metadata):
            fake_frame_settings = fake_frame_settings_factory(numeric_meta, input_path, None, transform)
            mock_file_settings.append(fake_frame_settings)

        image_dataset = id.ImageMapDataset(
            files=mock_file_settings, copy_preloaded_data=True, missing_std_mode=MissingStdMode.NONE,
            missing_std_value=0.0
        )

        return_values = []

        with patch("clair_torch.datasets.image_dataset.load_image") as mock:
            mock.return_value = val_image
            for i in range(len(image_dataset)):
                return_values.append(image_dataset[i])

        for val, std, numeric in return_values:
            assert torch.allclose(val_image, val)
            assert std is None

        assert return_values[0][2]['exposure_time'] == 0.5
        assert return_values[1][2]['exposure_time'] == 1.0
        assert return_values[2][2]['exposure_time'] == 3.0

    @pytest.mark.parametrize("std_mode, expected", [
        (MissingStdMode.NONE, None), (MissingStdMode.CONSTANT, torch.ones((2, 2))),
        (MissingStdMode.MULTIPLIER, 2 * torch.ones((2, 2)))
    ])
    def test_image_map_dataset_missing_std_mode_returns_expected(self, tmp_path, fake_frame_settings_factory, std_mode,
                                                                 expected):

        fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0}, tmp_path / "file1.tif", None,
                                                          MagicMock())

        image_dataset = id.ImageMapDataset([fake_frame_settings], missing_std_mode=std_mode, missing_std_value=1.0)

        with patch("clair_torch.datasets.image_dataset.load_image", return_value=2 * torch.ones((2, 2))):
            _, std_img, _ = image_dataset[0]

        if expected is None:
            assert std_img is None
        else:
            assert torch.allclose(std_img, expected)

    @pytest.mark.parametrize("copy_preloaded_data", [True, False])
    def test_image_map_dataset_preload_dataset(self, tmp_path, fake_frame_settings_factory, copy_preloaded_data):

        fake_frame_settings_1 = fake_frame_settings_factory({"exposure_time": 1.0}, tmp_path / "file1.tif",
                                                            tmp_path / "file1_std.tif", MagicMock())
        fake_frame_settings_2 = fake_frame_settings_factory({"exposure_time": 2.0}, tmp_path / "file2.tif",
                                                            tmp_path / "file2_std.tif", MagicMock())

        image_dataset = id.ImageMapDataset([fake_frame_settings_1, fake_frame_settings_2],
                                           copy_preloaded_data=copy_preloaded_data)

        return_tensor = torch.ones((2, 2))

        with patch("clair_torch.datasets.image_dataset.load_image", return_value=return_tensor):
            image_dataset.preload_dataset()

        val_0, std_0, meta_0 = image_dataset[0]
        val_1, std_1, meta_1 = image_dataset[1]

        if copy_preloaded_data:
            assert val_0.data_ptr() != return_tensor.data_ptr()
            assert val_1.data_ptr() != return_tensor.data_ptr()
            assert std_0.data_ptr() != return_tensor.data_ptr()
            assert std_1.data_ptr() != return_tensor.data_ptr()
        else:
            assert val_0.data_ptr() == return_tensor.data_ptr()
            assert val_1.data_ptr() == return_tensor.data_ptr()
            assert std_0.data_ptr() == return_tensor.data_ptr()
            assert std_1.data_ptr() == return_tensor.data_ptr()


class TestArtefactMapDataset:

    def test_artefact_map_dataset_match_found(self, tmp_path, fake_frame_settings_factory):

        main_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0, "magnification": 5.0},
                                                               tmp_path / "file1.tif",
                                                               tmp_path / "file1_std.tif", MagicMock())

        artefact_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 3.0, "magnification": 5.0},
                                                                   tmp_path / "file2.tif",
                                                                   tmp_path / "file2_std.tif", MagicMock())

        return_tensor = torch.ones((2, 2))
        artefact_dataset = id.ArtefactMapDataset([artefact_fake_frame_settings])

        with patch("clair_torch.datasets.image_dataset.load_image", return_value=return_tensor):
            val_img, std_img, num_meta = artefact_dataset.get_matching_artefact_image(main_fake_frame_settings,
                                                                                      ["magnification"])

        assert return_tensor.data_ptr() == val_img.data_ptr()
        assert return_tensor.data_ptr() == std_img.data_ptr()
        assert num_meta["exposure_time"] == 3.0

    def test_artefact_map_dataset_match_not_found(self, tmp_path, fake_frame_settings_factory):

        main_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0, "magnification": 6.0},
                                                               tmp_path / "file1.tif",
                                                               tmp_path / "file1_std.tif", MagicMock())

        artefact_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 3.0, "magnification": 5.0},
                                                                   tmp_path / "file2.tif",
                                                                   tmp_path / "file2_std.tif", MagicMock())

        return_tensor = torch.ones((2, 2))
        artefact_dataset = id.ArtefactMapDataset([artefact_fake_frame_settings])

        with patch("clair_torch.datasets.image_dataset.load_image", return_value=return_tensor):
            val_img, std_img, num_meta = artefact_dataset.get_matching_artefact_image(main_fake_frame_settings,
                                                                                      ["magnification"])

        assert val_img is None
        assert std_img is None
        assert num_meta is None
