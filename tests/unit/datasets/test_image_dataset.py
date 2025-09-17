import pytest
from unittest.mock import MagicMock, patch

import torch
from typeguard import suppress_type_checks

import clair_torch.datasets.base
from clair_torch.datasets import image_dataset as id
from clair_torch.common.enums import MissingStdMode, MissingValMode, FlatFieldMode, DarkFieldMode


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

        mock_file_settings = tuple(mock_file_settings)

        with suppress_type_checks():
            image_dataset = id.ImageMapDataset(
                files=mock_file_settings, copy_preloaded_data=True, missing_std_mode=MissingStdMode.NONE,
                missing_std_value=0.0, default_get_item_key="raw", missing_val_mode=MissingValMode.ERROR
            )

        return_values = []

        load_image_patch = patch.object(clair_torch.datasets.base, clair_torch.datasets.base.load_image.__name__)

        with load_image_patch as mock:
            mock.return_value = val_image
            for i in range(len(image_dataset)):
                return_values.append(image_dataset[i])

        # We expect None stds now.
        for idx, val, std, numeric in return_values:
            assert torch.allclose(val_image, val)
            assert std is None

        # With raw default_get_item_key the order should be the same as the order of the mock file_settings objects.
        assert return_values[0][3]['exposure_time'] == 1.0
        assert return_values[1][3]['exposure_time'] == 3.0
        assert return_values[2][3]['exposure_time'] == 0.5

    @pytest.mark.parametrize("std_mode, expected", [
        (MissingStdMode.NONE, None), (MissingStdMode.CONSTANT, torch.ones((2, 2))),
        (MissingStdMode.MULTIPLIER, 2 * torch.ones((2, 2)))
    ])
    def test_image_map_dataset_missing_std_mode_returns_expected(self, tmp_path, fake_frame_settings_factory, std_mode,
                                                                 expected):

        fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0}, tmp_path / "file1.tif", None,
                                                          MagicMock())

        with suppress_type_checks():
            image_dataset = id.ImageMapDataset((fake_frame_settings,), copy_preloaded_data=True,
                                               missing_std_mode=std_mode, missing_std_value=1.0,
                                               default_get_item_key="raw", missing_val_mode=MissingValMode.ERROR)

        load_image_patch = patch.object(clair_torch.datasets.base, clair_torch.datasets.base.load_image.__name__,
                                        return_value=2 * torch.ones((2, 2)))

        with load_image_patch:
            _, _, std_img, _ = image_dataset[0]

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

        with suppress_type_checks():
            image_dataset = id.ImageMapDataset((fake_frame_settings_1, fake_frame_settings_2),
                                               copy_preloaded_data=copy_preloaded_data,
                                               missing_val_mode=MissingValMode.ERROR,
                                               missing_std_mode=MissingStdMode.CONSTANT, missing_std_value=0.0,
                                               default_get_item_key="raw"
                                               )

        return_tensor = torch.ones((2, 2))

        load_image_patch = patch.object(clair_torch.datasets.base, clair_torch.datasets.base.load_image.__name__,
                                        return_value=return_tensor)

        with load_image_patch:

            image_dataset.preload_dataset()

        idx_0, val_0, std_0, meta_0 = image_dataset[0]
        idx_1, val_1, std_1, meta_1 = image_dataset[1]

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


class TestFlatFieldArtefactMapDataset:

    def test_flat_field_artefact_init(self):
        ...

    def test_flat_field_artefact__get_matching_artefact_image(self, tmp_path, fake_frame_settings_factory):

        main_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0, "magnification": 5.0},
                                                               tmp_path / "file1.tif",
                                                               tmp_path / "file1_std.tif", MagicMock())

        artefact_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 3.0, "magnification": 5.0},
                                                                   tmp_path / "file2.tif",
                                                                   tmp_path / "file2_std.tif", MagicMock())

        return_tensor = torch.ones((2, 2))

        with suppress_type_checks():
            artefact_dataset = id.FlatFieldArtefactMapDataset((artefact_fake_frame_settings,), True)

        # Define patches for other functions / methods.
        getitem_patch = patch.object(clair_torch.datasets.base.MultiFileMapDataset, "__getitem__",
                                     return_value=(0, return_tensor, return_tensor, {"exposure_time": 3.0}))
        match_idx_patch = patch.object(clair_torch.datasets.image_dataset.FlatFieldArtefactMapDataset,
                                       clair_torch.datasets.image_dataset.FlatFieldArtefactMapDataset._get_matching_image_settings_idx.__name__,
                                       return_value=0)

        with (getitem_patch, match_idx_patch):
            idx, val_img, std_img, num_meta = artefact_dataset._get_matching_artefact_image(main_fake_frame_settings)

        assert return_tensor.data_ptr() == val_img.data_ptr()
        assert return_tensor.data_ptr() == std_img.data_ptr()
        assert num_meta["exposure_time"] == 3.0

    @pytest.mark.parametrize("missing_val_mode, expected",
                             [(MissingValMode.ERROR, None), (MissingValMode.SKIP_BATCH, (None,) * 4)])
    def test_flat_field_artefact_get_matching_artefact_images_no_match(self, tmp_path, fake_frame_settings_factory,
                                                                       missing_val_mode, expected):

        main_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0, "magnification": 6.0},
                                                               tmp_path / "file1.tif",
                                                               tmp_path / "file1_std.tif", MagicMock())

        artefact_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 3.0, "magnification": 5.0},
                                                                   tmp_path / "file2.tif",
                                                                   tmp_path / "file2_std.tif", MagicMock())

        with suppress_type_checks():
            artefact_dataset = id.FlatFieldArtefactMapDataset((artefact_fake_frame_settings,), True,
                                                              missing_val_mode=missing_val_mode)

        get_matcing_artefact_image_patch = patch.object(clair_torch.datasets.base.MultiFileArtefactMapDataset,
                                                        clair_torch.datasets.base.MultiFileArtefactMapDataset._get_matching_artefact_image.__name__,
                                                        return_value=None)

        with (get_matcing_artefact_image_patch, suppress_type_checks()):

            if missing_val_mode == MissingValMode.ERROR:
                with pytest.raises(RuntimeError):
                    idx, val_img, std_img, num_meta = artefact_dataset.get_matching_artefact_images(
                        [main_fake_frame_settings])
            else:
                idx, val_img, std_img, num_meta = artefact_dataset.get_matching_artefact_images(
                    [main_fake_frame_settings])

        if missing_val_mode == MissingValMode.SKIP_BATCH:
            assert idx is None
            assert val_img is None
            assert std_img is None
            assert num_meta is None

    def test_flat_field_artefact_get_matching_artefact_images_matched(self, tmp_path, fake_frame_settings_factory):

        main_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 1.0, "magnification": 6.0},
                                                               tmp_path / "file1.tif",
                                                               tmp_path / "file1_std.tif", MagicMock())

        artefact_fake_frame_settings = fake_frame_settings_factory({"exposure_time": 3.0, "magnification": 5.0},
                                                                   tmp_path / "file2.tif",
                                                                   tmp_path / "file2_std.tif", MagicMock())

        with suppress_type_checks():
            artefact_dataset = id.FlatFieldArtefactMapDataset((artefact_fake_frame_settings,), True,
                                                              missing_val_mode=MissingValMode.ERROR)

        get_matcing_artefact_image_patch = patch.object(clair_torch.datasets.base.MultiFileArtefactMapDataset,
                                                        clair_torch.datasets.base.MultiFileArtefactMapDataset._get_matching_artefact_image.__name__,
                                                        return_value="dummy 0")
        collate_patch = patch.object(clair_torch.datasets.base, clair_torch.datasets.base.custom_collate.__name__,
                                     return_value="dummy 1")

        with (get_matcing_artefact_image_patch, collate_patch, suppress_type_checks()):

            ret = artefact_dataset.get_matching_artefact_images([main_fake_frame_settings])

        assert ret == "dummy 1"

