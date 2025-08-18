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


class TestTorchToCv:

    def test_torch_to_cv(self):

        with patch("clair_torch.common.transforms.torch_to_cv") as mock:

            x = torch.ones((2, 2))
            transform = tr.TorchToCv()
            _ = transform(x)

        mock.assert_called_once_with(x)


class TestBadPixelCorrection:

    def test_bad_pixel_correction(self):

        bad_pixel_map = torch.ones((2, 2))
        threshold = 1.0
        kernel_size = 1

        with patch("clair_torch.common.transforms.conditional_gaussian_blur") as mock:

            transform = tr.BadPixelCorrection(bad_pixel_map, threshold, kernel_size)
            _ = transform(bad_pixel_map)

        mock.assert_called_once_with(bad_pixel_map, bad_pixel_map, threshold, kernel_size)

    def test_bad_pixel_correction_invalid_args(self):

        good_map = torch.ones((2, 2))
        bad_map = "this is bad"
        good_threshold = 0.5
        bad_threshold = "this is bad"
        good_kernel_size = 2
        bad_kernel_size = "this is bad"

        combinations = [
            [bad_map, good_threshold, good_kernel_size],
            [good_map, bad_threshold, good_kernel_size],
            [good_map, good_threshold, bad_kernel_size]
        ]

        for combination in combinations:
            with pytest.raises(TypeError):

                arg1, arg2, arg3 = combination
                _ = tr.BadPixelCorrection(arg1, arg2, arg3)

        with pytest.raises(ValueError):

            _ = tr.BadPixelCorrection(good_map, good_threshold, 0)

    def test_bad_pixel_correction_clear_memory(self):

        pixel_map = torch.ones((2, 2))
        threshold = 0.5
        kernel_size = 1

        transform = tr.BadPixelCorrection(pixel_map, threshold, kernel_size)

        assert torch.allclose(pixel_map, transform.bad_pixel_map)
        transform.clear_memory()
        assert transform.bad_pixel_map is None


class TestStridedDownScale:

    def test_strided_downscale(self):

        transform = tr.StridedDownScale(step_size=2)

        assert transform.step_size == 2

    def test_strided_downscale_invalid_args(self):

        bad_type = "this is bad"
        too_small = -1

        with pytest.raises(TypeError):
            _ = tr.StridedDownScale(bad_type)
        with pytest.raises(ValueError):
            _ = tr.StridedDownScale(too_small)


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

    def test_cast_to_invalid_args(self):

        good_dtype = torch.float32
        bad_dtype = "this is bad"
        good_device = torch.device("cpu")
        bad_device = "this is bad"

        combinations = [
            [bad_dtype, good_device], [good_dtype, bad_device]
        ]

        for combination in combinations:
            with pytest.raises(TypeError):
                arg1, arg2 = combination
                _ = tr.CastTo(arg1, arg2)


class TestNormalize:

    def test_normalize(self):

        input_tensor = torch.ones((2, 2))

        with patch("clair_torch.common.transforms.normalize_tensor") as mock:

            normalize = tr.Normalize(min_val=0.0, max_val=1.0)
            _ = normalize(input_tensor)

        assert normalize.min_val == 0.0
        assert normalize.max_val == 1.0
        mock.assert_called_once_with(input_tensor, min_val=0.0, max_val=1.0, target_range=(0.0, 1.0))


class TestClampAlongDims:

    def test_clamp_along_dims(self):

        dim = 1
        min_max_pairs = (0, 1)
        x = torch.ones((2, 2))

        with patch("clair_torch.common.transforms.clamp_along_dims") as mock:

            transform = tr.ClampAlongDims(dim, min_max_pairs)
            _ = transform(x)

        mock.assert_called_once_with(x, dim, min_max_pairs)
