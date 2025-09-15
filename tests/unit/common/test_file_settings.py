import pytest
from unittest.mock import MagicMock, create_autospec, patch, ANY

import numpy as np

from clair_torch.common import file_settings as fs
from clair_torch.metadata.base import BaseMetadata


class TestFileSettingsInitialization:

    def test_filesettings_init_only_input_path(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        output_path = None
        default_output_root = None

        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=output_path,
                                        default_output_root=default_output_root,
                                        cpu_transforms=None)

        assert file_settings.input_path == input_path
        assert file_settings.default_output_root == input_path.parent / "clair_torch_output"
        assert file_settings.output_path == input_path.parent / "clair_torch_output" / "dummy_input.txt"

    def test_filesettings_init_input_path_and_output_path(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        output_path = tmp_path / "dummy_output.txt"

        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=output_path,
                                        default_output_root=None,
                                        cpu_transforms=None)

        assert file_settings.input_path == input_path
        assert file_settings.output_path == output_path
        assert file_settings.default_output_root == input_path.parent / "clair_torch_output"

    def test_filesettings_init_input_path_and_default_output_root(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        default_output_root = tmp_path / "dummy_root"

        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=None,
                                        default_output_root=default_output_root,
                                        cpu_transforms=None)

        assert file_settings.input_path == input_path
        assert file_settings.output_path == default_output_root / "dummy_input.txt"
        assert file_settings.default_output_root == default_output_root

    def test_filesettings_init_single_transform(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))

        mock_transform = MagicMock()

        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=None,
                                        default_output_root=None,
                                        cpu_transforms=mock_transform)

        assert isinstance(file_settings.cpu_transforms, list)
        assert len(file_settings.cpu_transforms) == 1
        assert mock_transform in file_settings.cpu_transforms
        mock_transform.assert_not_called()

    def test_filesettings_init_multiple_transforms(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))

        mock_transform_1 = MagicMock()
        mock_transform_2 = MagicMock()
        mock_transforms = [mock_transform_1, mock_transform_2]

        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=None,
                                        default_output_root=None,
                                        cpu_transforms=mock_transforms)

        assert isinstance(file_settings.cpu_transforms, list)
        assert len(file_settings.cpu_transforms) == 2
        for mock_transform in mock_transforms:
            mock_transform.assert_not_called()
            assert mock_transform in file_settings.cpu_transforms

    def test_filesettings_init_bad_input_path(self):
        input_path = 111

        with pytest.raises(TypeError, match=f"Expected path as str or Path, got {type(input_path).__name__}"):
            file_settings = fs.FileSettings(input_path=input_path)

    def test_filesettings_init_bad_output_path(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        output_path = 111

        with pytest.raises(TypeError, match=f"Expected path as str or Path, got {type(output_path).__name__}"):
            file_settings = fs.FileSettings(input_path=input_path, output_path=output_path)

    def test_filesettings_init_bad_default_output_root(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        default_output_root = 111

        with pytest.raises(TypeError, match=f"Expected path as str or Path, got {type(default_output_root).__name__}"):
            file_settings = fs.FileSettings(input_path=input_path, output_path=None,
                                            default_output_root=default_output_root)


class TestFileSettingsGetters:

    def test_file_settings_get_candidate_std_output_path(self, tmp_path):
        input_path = tmp_path / "dummy_input.txt"
        np.savetxt(input_path, np.zeros((2, 2)))
        file_settings = fs.FileSettings(input_path=input_path,
                                        output_path=None,
                                        default_output_root=None,
                                        cpu_transforms=None)

        expected_candidate_std_output_path = tmp_path / "clair_torch_output" / "dummy_input STD.txt"

        return_value = file_settings.get_candidate_std_output_path()

        assert expected_candidate_std_output_path == return_value


class TestFrameSettings:

    @pytest.mark.parametrize("numpy_array", [
        np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)
    ])
    def test_frame_settings_init_mock_metadata(self, numpy_array, temp_image_file):
        img_path = temp_image_file(numpy_array)
        mock_metadata = MagicMock()

        with patch('clair_torch.common.file_settings.issubclass', return_value=True):
            _ = fs.FrameSettings(input_path=img_path,
                                 output_path=None,
                                 default_output_root=None,
                                 metadata_cls=mock_metadata)

        mock_metadata.assert_called_once_with(img_path)

    @pytest.mark.parametrize("numpy_array", [
        np.tile([0.1, 0.2, 0.3], (256, 256, 1)).astype(np.float32)
    ])
    def test_frame_settings_metadata_getters(self, numpy_array, temp_image_file):
        img_path = temp_image_file(numpy_array)

        dummy_attributes = ["ree", "wee"]
        # Create a mock metadata instance and configure its methods
        mock_metadata_instance = MagicMock()
        mock_reference = create_autospec(BaseMetadata, instance=True)
        mock_metadata_instance.get_numeric_metadata.return_value = {"dummy number": 5}
        mock_metadata_instance.get_text_metadata.return_value = {"dummy text": "ree"}
        mock_metadata_instance.get_all_metadata.return_value = {"just dummy": 11}
        mock_metadata_instance.is_match.return_value = True

        # Create a mock class that returns the instance when called
        mock_metadata_cls = MagicMock(return_value=mock_metadata_instance)

        # Pass the mock class into FrameSettings
        with patch('clair_torch.common.file_settings.issubclass', return_value=True):
            frame_settings = fs.FrameSettings(input_path=img_path,
                                              output_path=None,
                                              default_output_root=None,
                                              metadata_cls=mock_metadata_cls)

        _ = frame_settings.get_numeric_metadata()
        _ = frame_settings.get_text_metadata()
        _ = frame_settings.get_all_metadata()
        _ = frame_settings.is_match(reference=mock_reference, attributes=dummy_attributes)

        # Assert the class was called once with the path
        mock_metadata_cls.assert_called_once_with(img_path)

        # Assert the methods of the instance were called once
        mock_metadata_instance.get_numeric_metadata.assert_called_once()
        mock_metadata_instance.get_text_metadata.assert_called_once()
        mock_metadata_instance.get_all_metadata.assert_called_once()
        mock_metadata_instance.is_match.assert_called_once_with(mock_reference, dummy_attributes)

    def test_frame_settings_get_numeric_metadata(self, frame_settings_with_mock_metadata):
        frame_settings, mock_instance, mock_cls, img_path = frame_settings_with_mock_metadata
        mock_instance.get_numeric_metadata.return_value = {"dummy": 42}

        _ = frame_settings.get_numeric_metadata()

        mock_cls.assert_called_once_with(img_path)
        mock_instance.get_numeric_metadata.assert_called_once()

    def test_frame_settings_get_text_metadata(self, frame_settings_with_mock_metadata):
        frame_settings, mock_instance, mock_cls, img_path = frame_settings_with_mock_metadata
        mock_instance.get_text_metadata.return_value = {"dummy": "42"}

        _ = frame_settings.get_text_metadata()

        mock_cls.assert_called_once_with(img_path)
        mock_instance.get_text_metadata.assert_called_once()

    def test_frame_settings_get_all_metadata(self, frame_settings_with_mock_metadata):
        frame_settings, mock_instance, mock_cls, img_path = frame_settings_with_mock_metadata
        mock_instance.get_all_metadata.return_value = {"dummy": "42"}

        _ = frame_settings.get_all_metadata()

        mock_cls.assert_called_once_with(img_path)
        mock_instance.get_all_metadata.assert_called_once()

    def test_frame_settings_is_match_mock_metadata(self, frame_settings_with_mock_metadata):
        frame_settings, mock_instance, mock_cls, img_path = frame_settings_with_mock_metadata
        mock_instance.is_match.return_value = True

        _ = frame_settings.get_all_metadata()

        mock_cls.assert_called_once_with(img_path)
        mock_instance.get_all_metadata.assert_called_once()


class TestPairedFileSettings:

    def test_paired_file_settings_init_val_input_path_only_no_std_file(self, temp_image_file):
        """
        Without std_input_path given, the init should fail with a FileNotFoundError.
        """
        val_input_path = temp_image_file()

        with pytest.raises(FileNotFoundError):
            _ = fs.PairedFileSettings(val_input_path=val_input_path)

    def test_paired_file_settings_init_val_input_path_implicit_std_file(self, temp_image_file):
        """
        Without std_input_path given, the init should fail with a FileNotFoundError. This tests implicit seeking of STD
        files, where the STD filepath isn't explicitly given, but a file with STD file candidate name pattern is found
        in the same directory as the value file.
        """
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="val_img STD.tif")  # Generate but don't use the std file in test.

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path)

        assert paired_file_settings.val_settings.input_path == val_input_path
        assert paired_file_settings.std_settings.input_path == std_input_path

    def test_paired_file_settings_init_val_and_std_input_paths(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path,
                                                     std_input_path=std_input_path)

        assert isinstance(paired_file_settings.val_settings, fs.FileSettings)
        assert isinstance(paired_file_settings.std_settings, fs.FileSettings)
        assert paired_file_settings.val_settings.input_path == val_input_path
        assert paired_file_settings.std_settings.input_path == std_input_path

    def test_paired_file_settings_init_cpu_transforms(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")
        val_mock_transform_1 = MagicMock()
        val_mock_transform_2 = MagicMock()
        std_mock_transform_1 = MagicMock()
        std_mock_transform_2 = MagicMock()
        val_transforms = [val_mock_transform_1, val_mock_transform_2]
        std_transforms = [std_mock_transform_1, std_mock_transform_2]

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path,
                                                     std_input_path=std_input_path,
                                                     val_cpu_transforms=val_transforms,
                                                     std_cpu_transforms=std_transforms)

        assert paired_file_settings.val_settings.cpu_transforms == val_transforms
        assert paired_file_settings.std_settings.cpu_transforms == std_transforms

    def test_paired_file_settings_get_input_paths(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path,
                                                     std_input_path=std_input_path)

        input_paths = paired_file_settings.get_input_paths()

        assert val_input_path == input_paths[0]
        assert std_input_path == input_paths[1]

    def test_paired_file_settings_get_output_paths(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")
        val_output_path = val_input_path.parent / "val_out.tif"
        std_output_path = std_input_path.parent / "std_out.tif"

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path,
                                                     std_input_path=std_input_path,
                                                     val_output_path=val_output_path,
                                                     std_output_path=std_output_path)

        output_paths = paired_file_settings.get_output_paths()

        assert val_output_path == output_paths[0]
        assert std_output_path == output_paths[1]

    def test_paired_file_settings_get_transforms(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")
        val_mock_transform_1 = MagicMock()
        val_mock_transform_2 = MagicMock()
        std_mock_transform_1 = MagicMock()
        std_mock_transform_2 = MagicMock()
        val_transforms = [val_mock_transform_1, val_mock_transform_2]
        std_transforms = [std_mock_transform_1, std_mock_transform_2]

        paired_file_settings = fs.PairedFileSettings(val_input_path=val_input_path,
                                                     std_input_path=std_input_path,
                                                     val_cpu_transforms=val_transforms,
                                                     std_cpu_transforms=std_transforms)

        transforms = paired_file_settings.get_transforms()

        assert val_transforms == transforms[0]
        assert std_transforms == transforms[1]


class TestPairedFrameSettings:

    def test_paired_frame_settings_init_val_input_path_and_std_input_path_only_val_metadata(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")

        dummy_attributes = ["ree", "wee"]
        # Create a mock metadata instance and configure its methods
        mock_metadata_instance = MagicMock()
        mock_reference = create_autospec(BaseMetadata, instance=True)
        mock_metadata_instance.get_numeric_metadata.return_value = {"dummy number": 5}
        mock_metadata_instance.get_text_metadata.return_value = {"dummy text": "ree"}
        mock_metadata_instance.get_all_metadata.return_value = {"just dummy": 11}
        mock_metadata_instance.is_match.return_value = True

        # Create a mock class that returns the instance when called
        mock_metadata_cls = MagicMock(return_value=mock_metadata_instance)

        # Pass the mock class into FrameSettings
        with patch('clair_torch.common.file_settings.issubclass', return_value=True):
            frame_settings = fs.PairedFrameSettings(val_input_path=val_input_path,
                                                    std_input_path=std_input_path,
                                                    metadata_cls=mock_metadata_cls)

        _ = frame_settings.get_numeric_metadata()
        _ = frame_settings.get_text_metadata()
        _ = frame_settings.get_all_metadata()
        _ = frame_settings.is_match(reference=mock_reference, attributes=dummy_attributes)

        assert frame_settings.std_metadata is None
        # Assert the class was called once with the path
        mock_metadata_cls.assert_called_once_with(val_input_path)

        # Assert the methods of the instance were called once
        mock_metadata_instance.get_numeric_metadata.assert_called_once()
        mock_metadata_instance.get_text_metadata.assert_called_once()
        mock_metadata_instance.get_all_metadata.assert_called_once()
        mock_metadata_instance.is_match.assert_called_once_with(mock_reference, dummy_attributes)

    def test_paired_frame_settings_init_val_input_path_and_std_input_path_both_metadata(self, temp_image_file):
        val_input_path = temp_image_file(name="val_img.tif")
        std_input_path = temp_image_file(name="this STD.tif")

        mock_metadata = MagicMock()

        mock_metadata_cls = MagicMock(return_value=mock_metadata)

        with patch('clair_torch.common.file_settings.issubclass', return_value=True):
            paired_frame_settings = fs.PairedFrameSettings(val_input_path=val_input_path,
                                                           std_input_path=std_input_path,
                                                           metadata_cls=mock_metadata_cls,
                                                           parse_std_meta=True)

        assert paired_frame_settings.std_metadata is not None
        mock_metadata_cls.assert_called_with(std_input_path)


class TestFileSettingsConstructor:

    def test_file_settings_constructor_class_usage_no_metadata(self, generate_temp_paired_files):
        paths = generate_temp_paired_files(num_paired=2, num_main=1, num_std=1, write_file=False)
        root_dir = paths[0].parent

        with patch("clair_torch.common.file_settings.FrameSettings") as MockFrame, \
                patch("clair_torch.common.file_settings.FileSettings") as MockFile, \
                patch("clair_torch.common.file_settings.PairedFrameSettings") as MockPairedFrame, \
                patch("clair_torch.common.file_settings.PairedFileSettings") as MockPairedFile, \
                patch("clair_torch.common.file_settings._get_file_input_paths_by_pattern", return_value=paths):
            mock_return_val = MagicMock()
            MockFrame.return_value = mock_return_val
            MockFile.return_value = mock_return_val
            MockPairedFrame.return_value = mock_return_val
            MockPairedFile.return_value = mock_return_val

            paired, main, std = fs.file_settings_constructor(
                dir_paths=root_dir,
                file_pattern=".tif",
                recursive=False,
            )

        assert MockFile.call_count == len(main) + len(std)
        assert MockPairedFile.call_count == len(paired)

    def test_file_settings_constructor_class_usage_with_metadata(self, generate_temp_paired_files):
        paths = generate_temp_paired_files(num_paired=2, num_main=1, num_std=1, write_file=False)
        root_dir = paths[0].parent
        mock_metadata = MagicMock()
        mock_metadata_cls = MagicMock(return_value=mock_metadata)

        with patch("clair_torch.common.file_settings.FrameSettings") as MockFrame, \
                patch("clair_torch.common.file_settings.FileSettings") as MockFile, \
                patch("clair_torch.common.file_settings.PairedFrameSettings") as MockPairedFrame, \
                patch("clair_torch.common.file_settings.PairedFileSettings") as MockPairedFile, \
                patch("clair_torch.common.file_settings._get_file_input_paths_by_pattern", return_value=paths):
            mock_return_val = MagicMock()
            MockFrame.return_value = mock_return_val
            MockFile.return_value = mock_return_val
            MockPairedFrame.return_value = mock_return_val
            MockPairedFile.return_value = mock_return_val

            paired, main, std = fs.file_settings_constructor(
                dir_paths=root_dir,
                file_pattern=".tif",
                recursive=False,
                metadata_cls=mock_metadata_cls
            )

        assert MockFrame.call_count == len(main) + len(std)
        assert MockPairedFrame.call_count == len(paired)

    def test_file_settings_constructor_parameter_passing(self, generate_temp_paired_files):
        paths = generate_temp_paired_files(num_paired=2, num_main=1, num_std=1, write_file=False)
        root_dir = paths[0].parent
        mock_metadata = MagicMock()
        mock_metadata_cls = MagicMock(return_value=mock_metadata)
        val_transform = MagicMock()
        std_transform = MagicMock()
        file_pattern = "*.tif"
        recursive = False
        dummy_out_root = paths[0].parent / "dummy_out_root"
        with patch("clair_torch.common.file_settings.FrameSettings") as MockFrame, \
                patch("clair_torch.common.file_settings.FileSettings") as MockFile, \
                patch("clair_torch.common.file_settings.PairedFrameSettings") as MockPairedFrame, \
                patch("clair_torch.common.file_settings.PairedFileSettings") as MockPairedFile, \
                patch("clair_torch.common.file_settings._get_file_input_paths_by_pattern",
                      return_value=paths) as mock_file_getter:
            mock_return_val = MagicMock()
            MockFrame.return_value = mock_return_val
            MockFile.return_value = mock_return_val
            MockPairedFrame.return_value = mock_return_val
            MockPairedFile.return_value = mock_return_val

            _, _, _ = fs.file_settings_constructor(
                dir_paths=root_dir,
                file_pattern=file_pattern,
                recursive=recursive,
                metadata_cls=mock_metadata_cls,
                val_cpu_transforms=val_transform,
                std_cpu_transforms=std_transform,
                default_output_root=dummy_out_root
            )

        mock_file_getter.assert_called_once_with(root_dir, file_pattern, recursive)
        MockFrame.assert_any_call(input_path=ANY, cpu_transforms=val_transform,
                                  metadata_cls=mock_metadata_cls, default_output_root=dummy_out_root)
        MockFrame.assert_any_call(input_path=ANY, cpu_transforms=std_transform,
                                  metadata_cls=mock_metadata_cls, default_output_root=dummy_out_root)
        MockPairedFrame.assert_any_call(val_input_path=ANY, std_input_path=ANY, default_output_root=dummy_out_root,
                                        val_cpu_transforms=val_transform, std_cpu_transforms=std_transform,
                                        metadata_cls=mock_metadata_cls)


class TestGroupFrameSettingsByAttributes:

    def test_group_frame_settings_by_attributes_single_attribute(self, fake_frame_settings_factory):

        frames = [
            fake_frame_settings_factory({'exposure': 10, 'gain': 1}),
            fake_frame_settings_factory({'exposure': 20, 'gain': 1}),
            fake_frame_settings_factory({'exposure': 10, 'gain': 2}),
        ]

        with patch("clair_torch.common.file_settings.validate_all", return_value=True):
            grouped = fs.group_frame_settings_by_attributes(frames, attributes={'exposure': None})

        assert len(grouped) == 2
        for group_meta, group_frames in grouped:
            assert all(f.metadata.values['exposure'] == group_meta['exposure'] for f in group_frames)

    def test_group_frame_settings_by_attributes_multiple_attributes(self, fake_frame_settings_factory):

        frames = [
            fake_frame_settings_factory({'exposure': 10, 'gain': 1, 'magnification': 5}),
            fake_frame_settings_factory({'exposure': 20, 'gain': 1, 'magnification': 5}),
            fake_frame_settings_factory({'exposure': 10, 'gain': 2, 'magnification': 5}),
            fake_frame_settings_factory({'exposure': 20, 'gain': 2, 'magnification': 5}),
            fake_frame_settings_factory({'exposure': 20, 'gain': 2, 'magnification': 10})
        ]

        with patch("clair_torch.common.file_settings.validate_all", return_value=True):
            grouped = fs.group_frame_settings_by_attributes(frames, attributes={'exposure': None, 'magnification': None})

        assert len(grouped) == 3
        for group_meta, group_frames in grouped:
            assert all(f.metadata.values['exposure'] == group_meta['exposure']
                       and f.metadata.values['magnification'] == group_meta['magnification'] for f in group_frames)

    def test_group_frame_settings_by_attributes_invalid_list_of_frame_settings(self):

        frames = [
            1, 2, 3
        ]

        with pytest.raises(TypeError):
            _ = fs.group_frame_settings_by_attributes(frames, "dummy")

    def test_group_frame_settings_by_attributes_invalid_attributes(self, fake_frame_settings_factory):

        frames = [
            1, 1
        ]

        with pytest.raises(TypeError):
            _ = fs.group_frame_settings_by_attributes(frames, 1)
