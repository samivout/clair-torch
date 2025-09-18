import pathlib
import re

import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from pathlib import Path

from typeguard import TypeCheckError
import numpy as np
import torch

import clair_torch.common.general_functions
from clair_torch.common.enums import ChannelOrder, DimensionOrder
import clair_torch.common.data_io as io


class TestLoadIcrfTxt:

    def test_load_icrf_txt_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="doesn't exist"):
            io.load_icrf_txt(Path("/non/existing/path.txt"))

    def test_load_icrf_txt_path_is_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Expected a filepath"):
            io.load_icrf_txt(tmp_path)  # tmp_path is a directory

    def test_load_icrf_txt_wrong_suffix(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("0.0, 0.1, 0.2")  # some dummy content
        with pytest.raises(ValueError, match="Expected .txt filetype"):
            io.load_icrf_txt(file)

    def test_load_icrf_txt_malformed_data(self, tmp_path):
        bad_txt = tmp_path / "bad.txt"
        bad_txt.write_text("this is not numeric data")
        with pytest.raises(IOError, match="Failed to load NumPy array"):
            io.load_icrf_txt(bad_txt)

    @pytest.mark.parametrize("temp_icrf_txt_file", [np.random.rand(256, 3)], indirect=True)
    def test_load_icrf_txt_invalid_channel_order(self, temp_icrf_txt_file):

        source_channel_order = "invalid channel order"

        with pytest.raises(TypeCheckError):
            io.load_icrf_txt(temp_icrf_txt_file, source_channel_order=source_channel_order)

    @pytest.mark.parametrize("temp_icrf_txt_file", [np.tile([0.1, 0.2, 0.3], (256, 1)).astype(np.float32)], indirect=True)
    @pytest.mark.parametrize("channel_order", [ChannelOrder.RGB, ChannelOrder.BGR, ChannelOrder.ANY])
    @pytest.mark.parametrize("dimension_order", [DimensionOrder.BSC, DimensionOrder.BCS])
    def test_load_icrf_txt_success(self, tmp_path, temp_icrf_txt_file, channel_order, dimension_order):

        reference_data_as_torch = torch.from_numpy(np.tile([0.1, 0.2, 0.3], (256, 1))).to(dtype=torch.float32)

        tensor = io.load_icrf_txt(temp_icrf_txt_file, source_channel_order=channel_order,
                                  source_dimension_order=dimension_order)

        if dimension_order == DimensionOrder.BSC:
            reference_data_as_torch = torch.transpose(reference_data_as_torch, 0, 1)

        if channel_order == ChannelOrder.BGR:
            reference_data_as_torch = reference_data_as_torch[[2, 1, 0], :]

        assert isinstance(tensor, torch.Tensor)
        assert reference_data_as_torch.shape == tensor.shape
        assert torch.allclose(reference_data_as_torch, tensor)


class TestLoadImage:

    @pytest.mark.parametrize("numpy_array", [np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)])
    def test_load_image_success_no_transform(self, tmp_path, temp_image_file, numpy_array):
        img_path = temp_image_file(numpy_array)

        reference_data_as_torch = torch.from_numpy(np.tile([0.1, 0.2, 0.3], (256, 256, 1))).to(dtype=torch.float32)
        reference_shape = reference_data_as_torch.shape

        image_tensor = io.load_image(img_path, transforms=None).to(torch.float32)

        assert isinstance(image_tensor, torch.Tensor)

        if len(reference_shape) == 3 and reference_shape[2] == 3:
            image_tensor = image_tensor[[2, 1, 0], ...]
            image_tensor = image_tensor.permute(2, 1, 0)
            assert reference_shape == image_tensor.shape
            assert torch.allclose(reference_data_as_torch, image_tensor)
        elif len(reference_shape) == 1:
            assert reference_shape == image_tensor.shape
            assert torch.allclose(reference_data_as_torch, image_tensor)

    @pytest.mark.parametrize("numpy_array", [
        np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)
    ])
    def test_load_image_success_with_transform(self, numpy_array, temp_image_file):
        img_path = temp_image_file(numpy_array)

        reference_data_as_torch = torch.from_numpy(numpy_array).to(dtype=torch.float32)

        mock_transform = MagicMock()
        mock_transform.return_value = torch.ones_like(reference_data_as_torch)

        with patch.object(clair_torch.common.data_io, clair_torch.common.data_io.normalize_container.__name__,
                          return_value=[mock_transform]):
            _ = io.load_image(img_path, transforms=mock_transform)

        mock_transform.assert_called_once()

    @pytest.mark.parametrize("numpy_array", [
        np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)
    ])
    def test_load_image_success_with_multiple_transforms(self, numpy_array, temp_image_file):
        img_path = temp_image_file(numpy_array)

        reference_data_as_torch = torch.from_numpy(numpy_array).to(dtype=torch.float32)

        mock_transform_1 = MagicMock()
        mock_transform_2 = MagicMock()
        mock_transform_1.return_value = torch.ones_like(reference_data_as_torch)
        mock_transform_2.return_value = torch.ones_like(reference_data_as_torch)

        with patch.object(clair_torch.common.data_io, clair_torch.common.data_io.normalize_container.__name__,
                          return_value=[mock_transform_1, mock_transform_2]):
            _ = io.load_image(img_path, transforms=(mock_transform_1, mock_transform_2))

        mock_transform_1.assert_called_once()
        mock_transform_2.assert_called_once()

    def test_load_image_invalid_transform(self):

        dummy_image = np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)

        class FakeTransform:
            pass

        with (patch.object(clair_torch.common.data_io.cv, clair_torch.common.data_io.cv.imread.__name__, return_value=dummy_image),
              patch.object(pathlib.Path, pathlib.Path.exists.__name__, return_value=True),
              patch.object(pathlib.Path, pathlib.Path.is_file.__name__, return_value=True)):

            transforms = [FakeTransform()]

            with pytest.raises(TypeError):
                _ = io.load_image(Path("fake/path"), transforms=transforms)

    def test_load_image_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="doesn't exist"):
            io.load_image(Path("/non/existing/path.txt"))

    def test_load_image_path_is_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Expected a filepath"):
            io.load_icrf_txt(tmp_path)  # tmp_path is a directory

    def test_load_image_malformed_data(self, tmp_path):
        bad_txt = tmp_path / "bad.txt"
        bad_txt.write_text("this is not numeric data")
        with pytest.raises(IOError, match="Failed to load NumPy array"):
            io.load_image(bad_txt)


class TestSaveImage:

    def test_save_image_invalid_image(self, tmp_path):
        image_save_path = tmp_path / "image.tif"
        bad_dummy_image = "this is bad"
        with pytest.raises(TypeCheckError):
            io.save_image(bad_dummy_image, image_save_path)

    def test_save_image_invalid_path(self):
        bad_dummy_path = 1
        dummy_image = torch.ones((5, 5), dtype=torch.uint8)
        with pytest.raises(TypeCheckError):
            io.save_image(dummy_image, bad_dummy_path)

    def test_save_image_bad_windows_path(self, tmp_path):
        bad_windows_path = tmp_path / ":_image.tif"
        dummy_image = torch.ones((5, 5), dtype=torch.uint8)
        with pytest.raises(IOError):
            io.save_image(dummy_image, bad_windows_path)

    def test_save_image_bad_opencv_parameters(self, tmp_path):
        image_save_path = tmp_path / "image.tif"
        dummy_image = torch.ones((5, 5), dtype=torch.uint8)
        params = ["this is bad"]
        with pytest.raises(TypeCheckError):
            io.save_image(dummy_image, image_save_path, params=params)

    @pytest.mark.parametrize(
        "tensor", [torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1).expand(3, 256, 256).clone().to(dtype=torch.float32),
                   torch.ones((256, 256)).to(dtype=torch.float32)]
    )
    def test_save_image_roundtrip(self, tmp_path, tensor):
        image_save_path = tmp_path / "image.tif"
        io.save_image(tensor, image_save_path)
        loaded_image = io.load_image(image_save_path).to(dtype=torch.float32)
        assert torch.allclose(loaded_image, tensor)


class TestSaveIcrfTxt:

    def test_save_icrf_txt_invalid_data(self, tmp_path):
        bad_tensor = "this is bad"
        save_path = tmp_path / "icrf.txt"

        with pytest.raises(TypeCheckError):
            io.save_icrf_txt(bad_tensor, save_path)

    def test_save_icrf_txt_invalid_path(self):
        icrf = torch.ones(3, 256)
        save_path = 111
        with pytest.raises(TypeCheckError):
            io.save_icrf_txt(icrf, save_path)

    @pytest.mark.parametrize("channel_order", [ChannelOrder.RGB, ChannelOrder.BGR, ChannelOrder.ANY])
    @pytest.mark.parametrize("dimension_order", [DimensionOrder.BSC, DimensionOrder.BCS])
    def test_save_icrf_txt_roundtrip(self, tmp_path, channel_order, dimension_order):
        icrf = torch.ones(3, 256) * 0.1
        save_path = tmp_path / "icrf.txt"

        io.save_icrf_txt(icrf, save_path, channel_order, dimension_order)
        loaded_icrf = io.load_icrf_txt(save_path, channel_order, dimension_order)

        assert torch.allclose(icrf, loaded_icrf)


class TestGetFileInputPathsByPattern:

    @pytest.mark.parametrize("path_type", [str, Path])
    def test_get_file_paths_recursive(self, generate_temp_file_tree, path_type):
        root = generate_temp_file_tree(depth=2, num_files_per_level=1, file_suffix=".tif")
        input_path = path_type(root)
        result = io._get_file_input_paths_by_pattern(input_path, "*.tif", recursive_search=True)
        assert len(result) == 2

    def test_get_file_paths_non_recursive(self, generate_temp_file_tree):
        root = generate_temp_file_tree(depth=2, num_files_per_level=1, file_suffix=".tif")
        result = io._get_file_input_paths_by_pattern(root, "*.tif", recursive_search=False)
        # Only top-level file
        assert len(result) == 1

    def test_get_file_paths_invalid_type(self):
        with pytest.raises(TypeCheckError):
            io._get_file_input_paths_by_pattern(123, "*.tif")

    def test_get_file_input_paths_by_type_invalid_pattern(self):
        with pytest.raises(TypeCheckError):
            io._get_file_input_paths_by_pattern("/dummy_path", 123)


class TestPairMainAndStdFiles:

    def test_pair_main_and_std_files_strict_mode(self, generate_temp_paired_files):
        paths = generate_temp_paired_files(num_paired=3, num_main=2, num_std=2)
        paired, main_only, std_only = io._pair_main_and_std_files(paths, strict_exclusive=True)

        assert len(paired) == 3
        assert len(main_only) == 2
        assert len(std_only) == 2

    def test_pair_main_and_std_files_nonstrict_mode(self, generate_temp_paired_files):
        paths = generate_temp_paired_files(num_paired=3, num_main=2, num_std=2)
        paired, main_only, std_only = io._pair_main_and_std_files(paths, strict_exclusive=False)

        assert len(paired) == 3
        assert len(main_only) == 5
        assert len(std_only) == 5

    def test_pair_main_and_std_files_empty_list(self):

        files = []

        paired, main_only, std_only = io._pair_main_and_std_files(files)

        assert len(paired) == 0
        assert len(main_only) == 0
        assert len(std_only) == 0


class TestLoadVideoFramesGenerator:

    def test_load_video_frames_generator_success_no_transforms(self, tmp_path):

        fake_frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        fake_frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 200

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (True, fake_frame1),
            (True, fake_frame2),
            (False, None)  # End of video
        ]

        video_capture_patch = patch.object(clair_torch.common.data_io.cv, clair_torch.common.data_io.cv.VideoCapture.__name__, return_value=mock_cap)
        validate_input_file_patch = patch.object(clair_torch.common.data_io, clair_torch.common.data_io.validate_input_file_path.__name__)

        with (video_capture_patch, validate_input_file_patch):

            frames = list(io.load_video_frames_generator(tmp_path, transforms=None))

        assert len(frames) == 2
        assert isinstance(frames[0], torch.Tensor)
        assert mock_cap.release.called
