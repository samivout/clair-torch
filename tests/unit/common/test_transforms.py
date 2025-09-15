import pytest
from unittest.mock import patch

import torch

import clair_torch.common.transforms as tr


class TestCvToTorch:

    def test_cv_to_torch(self):

        with patch("clair_torch.common.transforms.cv_to_torch") as mock:

            x = torch.ones((2, 2))
            transform = tr.CvToTorch()
            _ = transform(x)

        mock.assert_called_once_with(x)

    def test_cv_to_torch_to_config(self):

        transform = tr.CvToTorch()
        assert transform.to_config() == {}


class TestTorchToCv:

    def test_torch_to_cv(self):

        with patch("clair_torch.common.transforms.torch_to_cv") as mock:

            x = torch.ones((2, 2))
            transform = tr.TorchToCv()
            _ = transform(x)

        mock.assert_called_once_with(x)

    def test_torch_to_cv_to_config(self):

        transform = tr.TorchToCv()
        assert transform.to_config() == {}


class TestStridedDownScale:

    def test_strided_downscale(self):

        transform = tr.StridedDownscale(step_size=2)

        assert transform.step_size == 2

    def test_strided_downscale_invalid_args(self):

        bad_type = "this is bad"
        too_small = -1

        with pytest.raises(TypeError):
            _ = tr.StridedDownscale(bad_type)
        with pytest.raises(ValueError):
            _ = tr.StridedDownscale(too_small)

    def test_strided_downscale_config_roundtrip(self):

        transform = tr.StridedDownscale(step_size=2)
        expected = {"step_size": 2}

        assert transform.to_config() == expected

        transform_2 = tr.StridedDownscale.from_config(expected)

        assert transform.step_size == transform_2.step_size


class TestCastTo:

    def test_cast_to(self):

        if torch.cuda.is_available():
            dtype = torch.float64
            cpu_device = torch.device("cpu")
            gpu_device = torch.device("cuda", index=0)

            input_tensor = torch.zeros((2, 2), device=gpu_device, dtype=torch.float32)

            only_dtype = tr.CastTo(data_type=dtype)
            only_device = tr.CastTo(device=cpu_device)
            both = tr.CastTo(data_type=dtype, device=cpu_device)

            dtype_modified = only_dtype(input_tensor)
            assert dtype_modified.dtype == torch.float64
            assert dtype_modified.device == gpu_device

            device_modified = only_device(input_tensor)
            assert device_modified.dtype == torch.float32
            assert device_modified.device == cpu_device

            both_modified = both(input_tensor)
            assert both_modified.dtype == torch.float64
            assert both_modified.device == cpu_device

    def test_cast_to_invalid_device(self):

        good_dtype = torch.float32
        bad_device = "this is bad"

        with pytest.raises(RuntimeError):
            _ = tr.CastTo(good_dtype, bad_device)

    def test_cast_to_invalid_dtype(self):

        bad_dtype = "this is bad"
        good_device = torch.device("cpu")

        with pytest.raises(KeyError):
            _ = tr.CastTo(bad_dtype, good_device)

    def test_cast_to_config_roundtrip(self):

        transform = tr.CastTo(data_type=torch.float32, device=torch.device("cpu"))
        expected = {"data_type": "float32", "device": "cpu"}

        assert transform.to_config() == expected

        transform_2 = tr.CastTo.from_config(expected)

        assert transform.device == transform_2.device
        assert transform.data_type == transform_2.data_type


class TestNormalize:

    def test_normalize(self):

        input_tensor = torch.ones((2, 2))

        with patch("clair_torch.common.transforms.normalize_tensor") as mock:

            normalize = tr.Normalize(min_val=0.0, max_val=1.0)
            _ = normalize(input_tensor)

        assert normalize.min_val == 0.0
        assert normalize.max_val == 1.0
        mock.assert_called_once_with(input_tensor, min_val=0.0, max_val=1.0, target_range=(0.0, 1.0))

    def test_normalize_config_roundtrip(self):

        transform = tr.Normalize(min_val=None, max_val=None, target_range=(0.0, 1.0))
        expected = {"min_val": None, "max_val": None, "target_range": [0.0, 1.0]}

        assert transform.to_config() == expected

        transform_2 = tr.Normalize.from_config(expected)

        assert transform.min_val == transform_2.min_val
        assert transform.max_val == transform_2.max_val
        assert transform.target_range == transform_2.target_range


class TestClampAlongDims:

    def test_clamp_along_dims(self):

        dim = 1
        min_max_pairs = (0, 1)
        x = torch.ones((2, 2))

        with patch("clair_torch.common.transforms.clamp_along_dims") as mock:

            transform = tr.ClampAlongDims(dim, min_max_pairs)
            _ = transform(x)

        mock.assert_called_once_with(x, dim, min_max_pairs)

    @pytest.mark.parametrize("min_max_pairs, expected", [
        ((0.0, 1.0), {"dim": (1, 2), "min_max_pairs": (0.0, 1.0)}),
        ([(0.0, 1.0), (0.1, 0.9)], {"dim": (1, 2), "min_max_pairs": [(0.0, 1.0), (0.1, 0.9)]})
    ])
    def test_clamp_along_dims_config_roundtrip(self, min_max_pairs, expected):

        dim = (1, 2)
        transform = tr.ClampAlongDims(dim=dim, min_max_pairs=min_max_pairs)

        assert transform.to_config() == expected

        transform_2 = tr.ClampAlongDims.from_config(expected)

        assert transform.dim == transform_2.dim
        assert transform.min_max_pairs == transform_2.min_max_pairs


class TestSerializeTransforms:

    def test_serialize_transforms(self):

        expected = [{'params': 'dummy', 'type': 'Normalize'}, {'params': 'dummy', 'type': 'CvToTorch'}]

        with (patch("clair_torch.common.transforms.Normalize.to_config", return_value="dummy"),
              patch("clair_torch.common.transforms.CvToTorch.to_config", return_value="dummy")):

            transform_1 = tr.Normalize()
            transform_2 = tr.CvToTorch()

            serialized = tr.serialize_transforms([transform_1, transform_2])

            assert serialized == expected


class TestDeserializeTransforms:

    def test_deserialize_transforms(self):

        serialized = [{'params': 'dummy', 'type': 'Normalize'}, {'params': 'dummy', 'type': 'CvToTorch'}]

        with (patch("clair_torch.common.transforms.Normalize.from_config", return_value="dummy_1"),
              patch("clair_torch.common.transforms.CvToTorch.from_config", return_value="dummy_2")):

            deserialized = tr.deserialize_transforms(serialized)
            assert len(deserialized) == 2
            assert deserialized[0] == "dummy_1"
            assert deserialized[1] == "dummy_2"
