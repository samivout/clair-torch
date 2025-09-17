import pytest
from hypothesis import given, strategies as st
from hypothesis_torch import tensor_strategy

from typeguard import TypeCheckError
import numpy as np
import torch

from clair_torch.common import general_functions as gf


class TestNormalizeContainer:
    @pytest.mark.parametrize(
        "value,             target_type,    convert,    expected", [
            (None, list, False, None),
            (None, list, False, []),
            (5, list, False, [5]),
            ([1, 2], list, False, [1, 2]),
            ((1, 2), list, True, [1, 2]),
            ((1, 2), tuple, False, (1, 2)),
            ("abc", list, False, ["abc"]),
            (np.array([1, 2]), list, False, [np.array([1, 2])]),
        ])
    def test_normalize_container(self, value, target_type, convert, expected):
        none_if_all_none = expected is None
        result = gf.normalize_container(value, target_type, convert_if_iterable=convert,
                                        none_if_all_none=none_if_all_none)
        if expected is None:
            assert result is None
        elif isinstance(expected, (list, tuple)):
            if not expected:
                assert result == expected
            elif isinstance(expected[0], np.ndarray):
                np.testing.assert_equal(expected[0], result[0])
            else:
                assert list(result) == list(expected)
        else:
            assert result is expected

    def test_none_if_all_none_on_container_with_nones(self):
        assert gf.normalize_container([None, None], list) is None
        assert gf.normalize_container((None, None), tuple) is None
        assert gf.normalize_container([None, 1], list) == [None, 1]


class TestWeightedMeanAndStd:

    @pytest.mark.slow
    @pytest.mark.parametrize(("compute_std", "keepdim"), [(True, True), (False, False)])
    @given(ndim=st.integers(min_value=1, max_value=4), data=st.data())
    def test_weighted_mean_and_std(self, ndim, keepdim, compute_std, data, device_fixture):

        target_device = device_fixture
        reduction_dim_strategy = st.one_of(st.integers(min_value=0, max_value=ndim - 1), st.none())
        shared_shape = st.shared(st.lists(
            st.integers(min_value=1, max_value=25),
            min_size=ndim,
            max_size=ndim
        ), key="shape")

        value_strategy = tensor_strategy(
            dtype=torch.float32,
            device=target_device,
            shape=shared_shape,
            elements=st.floats(min_value=-1e3, max_value=1e3)
        )

        weight_strategy = tensor_strategy(
            dtype=torch.float32,
            device=target_device,
            shape=shared_shape,
            elements=st.floats(min_value=0, max_value=1e3)
        )

        bool_strategy = tensor_strategy(
            dtype=torch.float32,
            device=target_device,
            shape=shared_shape,
            elements=st.booleans()
        )

        reduction_dim = data.draw(reduction_dim_strategy)
        values = data.draw(value_strategy)
        weights = data.draw(st.one_of(weight_strategy, st.none()))
        mask = data.draw(st.one_of(bool_strategy, st.none()))

        mean, std = gf.weighted_mean_and_std(
            values=values,
            weights=weights,
            mask=mask,
            dim=reduction_dim,
            keepdim=keepdim,
            compute_std=compute_std
        )

        # Check STD computation is respected.
        assert isinstance(mean, torch.Tensor)
        if compute_std:
            assert isinstance(std, torch.Tensor)
            assert (std >= 0).all()
        else:
            assert std is None

        # Regression case, ignoring weights and masks the result should be close to
        if mask is None and weights is None:
            expected_mean = values.mean(dim=reduction_dim, keepdim=keepdim)
            assert torch.allclose(mean, expected_mean, atol=1e-4)

    def test_simple_mean_computation(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weights = None
        mask = None
        dim = 1
        keepdim = False
        compute_std = False

        mean, std = gf.weighted_mean_and_std(
            values=values,
            weights=weights,
            mask=mask,
            dim=dim,
            keepdim=keepdim,
            compute_std=compute_std
        )

        expected_mean = values.mean(dim=dim, keepdim=keepdim)
        assert torch.allclose(mean, expected_mean, atol=1e-6)
        assert std is None

    def test_weighted_mean_computation(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weights = torch.tensor([[1.0, 1.0], [0.25, 0.75]])
        mask = None
        dim = 1
        keepdim = False
        compute_std = False

        mean, std = gf.weighted_mean_and_std(
            values=values,
            weights=weights,
            mask=mask,
            dim=dim,
            keepdim=keepdim,
            compute_std=compute_std
        )

        expected_mean = (values * weights).sum(dim=dim) / weights.sum(dim=dim)
        assert torch.allclose(mean, expected_mean, atol=1e-6)
        assert std is None

    def test_masked_mean_computation(self):

        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [False, True]])  # Only include 1.0 and 4.0
        weights = None
        dim = 1
        keepdim = False
        compute_std = False

        mean, std = gf.weighted_mean_and_std(
            values=values,
            weights=weights,
            mask=mask,
            dim=dim,
            keepdim=keepdim,
            compute_std=compute_std
        )

        # Only include values where mask == True
        masked_values = values * mask.to(dtype=values.dtype)
        expected_mean = masked_values.sum(dim=dim) / mask.sum(dim=dim)

        assert torch.allclose(mean, expected_mean, atol=1e-6)
        assert std is None


class TestFlatFieldMean:

    def test_flat_field_mean(self):
        """
        Flat field mean function is tested by setting a certain area of zeros tensor to ones, and computing the flat
        field mean over that area, which should return 1.0.
        """
        N, C, H, W = 1, 1, 10, 10
        mid_area_side_fraction = 0.5

        flat_field = torch.zeros((N, C, H, W))
        flat_field[:, :, 2:7, 2:7] = 1.0

        mean = gf.flat_field_mean(flat_field, mid_area_side_fraction)

        expected = torch.ones((N, C, 1, 1))
        assert torch.allclose(mean, expected, atol=1e-12)

    def test_flat_field_mean_fractional_roi(self):
        """
        Similar to the base test_flat_field_mean() test, but testing with fractional ROI.
        """
        # Create a 1x1x6x6 tensor filled with zeros
        flat_field = torch.zeros((1, 1, 6, 6))

        flat_field[:, :, 2:4, 2:4] = 1.0

        expected_mean = 1.0

        result = gf.flat_field_mean(flat_field, mid_area_side_fraction=1 / 3)

        assert result.shape == (1, 1, 1, 1)
        assert torch.allclose(result, torch.tensor([[expected_mean]]), atol=1e-12)

    def test_flat_field_mean_invalid_tensor(self):

        flat_field = "this is bad"

        with pytest.raises(TypeCheckError):
            gf.flat_field_mean(flat_field, 0.5)

    def test_flat_field_mean_invalid_mid_area_side_fraction(self):

        flat_field = torch.ones((10, 10))
        mid_area_side_fraction = "more bad"

        with pytest.raises(TypeCheckError):
            gf.flat_field_mean(flat_field, mid_area_side_fraction)

    def test_flat_field_mean_mid_area_side_fraction_value_error(self):

        flat_field = torch.ones((10, 10))
        mid_area_side_fraction = 1.01

        with pytest.raises(ValueError):
            gf.flat_field_mean(flat_field, mid_area_side_fraction)


class TestFlatFieldCorrection:

    @pytest.mark.parametrize("images, flatfield, flatfield_mean_val, epsilon", [
        ("bad", torch.ones((1, 1)), torch.ones((1, 1)), 1e-8),
        (torch.ones((1, 1)), "bad", torch.ones((1, 1)), 1e-8),
        (torch.ones((1, 1)), torch.ones((1, 1)), "bad", 1e-8),
        (torch.ones((1, 1)), torch.ones((1, 1)), torch.ones((1, 1)), "bad"),
    ])
    def test_flat_field_correction_invalid_args(self, images, flatfield, flatfield_mean_val, epsilon):

        with pytest.raises(TypeCheckError):
            gf.flatfield_correction(images, flatfield, flatfield_mean_val, epsilon)

    def test_flat_field_correction_finite_gradients(self):

        multiplier = 10
        images = torch.ones((2, 3, 4, 4)) * multiplier
        flatfield = torch.zeros((1, 3, 4, 4))  # Zero flatfield as worst case.

        flatfield_mean = torch.ones((1, 3, 1, 1))
        epsilon = 1e-6
        expected_safe = multiplier / epsilon + 1

        corrected = gf.flatfield_correction(images, flatfield, flatfield_mean, epsilon=epsilon)

        assert torch.isfinite(corrected).all()
        assert (corrected < expected_safe).all()
        assert corrected.shape == images.shape

    @pytest.mark.parametrize(
        "image_shape, flatfield_shape, flatfield_mean_val_shape",
        [
            # Too many dimensions (e.g., (..., C, extra_dim))
            ((2, 3, 16, 16), (2, 3, 16, 16), (2, 3, 2)),
            # Too few dimensions (e.g., scalar instead of (C,) or (..., C))
            ((2, 3, 16, 16), (2, 3, 16, 16), ()),  # Scalar tensor
            # Not aligned with channel dimension (e.g., shape does not match (..., C))
            ((2, 3, 16, 16), (2, 3, 16, 16), (3, 16)),  # Not broadcastable to (..., C, 1, 1)
            ((2, 3, 16, 16), (2, 16, 16), (3, ))
        ]
    )
    def test_flatfield_correction_invalid_mean_shapes(self, image_shape, flatfield_shape, flatfield_mean_val_shape):
        def make_tensor(shape, fill_value=1.0):
            if len(shape) == 0:
                return torch.tensor(fill_value)
            return torch.full(shape, fill_value)

        images = make_tensor(image_shape)
        flatfield = make_tensor(flatfield_shape)
        flatfield_mean_val = make_tensor(flatfield_mean_val_shape)

        with pytest.raises(ValueError):
            gf.flatfield_correction(images, flatfield, flatfield_mean_val)


class TestGetValidExposurePairs:

    def test_get_valid_exposure_pairs_no_threshold(self):

        exposures = torch.tensor([1.0, 2.0, 4.0])

        i_idx, j_idx, ratios = gf.get_valid_exposure_pairs(exposures)

        expected_i = torch.tensor([0, 0, 1])
        expected_j = torch.tensor([1, 2, 2])
        expected_ratios = torch.tensor([0.5, 0.25, 0.5])  # i / j for (0,1), (0,2), (1,2)

        assert torch.equal(i_idx, expected_i)
        assert torch.equal(j_idx, expected_j)
        assert torch.allclose(ratios, expected_ratios)

    def test_get_valid_exposure_pairs_with_threshold(self):

        exposures = torch.tensor([1.0, 2.0, 4.0])
        threshold = 0.4

        i_idx, j_idx, ratios = gf.get_valid_exposure_pairs(exposures, exposure_ratio_threshold=threshold)

        expected_i = torch.tensor([0, 1])
        expected_j = torch.tensor([1, 2])
        expected_ratios = torch.tensor([0.5, 0.5])  # Only pairs (0,1) and (1,2) pass the threshold check.

        assert torch.equal(i_idx, expected_i)
        assert torch.equal(j_idx, expected_j)
        assert torch.allclose(ratios, expected_ratios)

    @pytest.mark.parametrize("increasing_exposure_values, exposure_ratio_threshold", [
        ("bad", 0.1), (torch.ones((5, )), "bad")
    ])
    def test_get_valid_exposure_pairs_invalid_args(self, increasing_exposure_values, exposure_ratio_threshold):

        with pytest.raises(TypeCheckError):
            gf.get_valid_exposure_pairs(increasing_exposure_values, exposure_ratio_threshold)


class TestGetPairwiseValidPixelMask:

    def test_get_valid_pixel_mask_without_std(self):

        val_lower, val_upper = 0.1, 1.0

        image_stack = torch.tensor([
            [[[0.1, 0.5], [0.9, 1.0]]],
            [[[0.2, 0.6], [0.4, 0.8]]],
            [[[0.05, 0.95], [1.1, 0.0]]]  # This one has values below val_lower and above val_upper
        ])
        i_idx = torch.tensor([0, 0, 1])
        j_idx = torch.tensor([1, 2, 2])

        mask = gf.get_pairwise_valid_pixel_mask(image_stack, i_idx, j_idx, val_lower=val_lower, val_upper=val_upper)

        # We expect:
        # Pair (0,1): All values are in range -> mask is all True
        # Pair (0,2): image 2 has invalid pixels -> some False
        # Pair (1,2): image 2 has invalid pixels -> some False

        expected_mask = torch.tensor([
            [[[True, True], [True, True]]],
            [[[False, True], [False, False]]],
            [[[False, True], [False, False]]],
        ])

        assert torch.equal(mask, expected_mask)

    def test_valid_pixel_mask_with_std(self):

        val_lower, val_upper, std_lower, std_upper = 0.0, 1.0, 0.0, 5e-4

        image_stack = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[0.5, 0.5], [0.5, 0.5]]],
        ])

        std_stack = torch.tensor([
            [[[1e-4, 6e-4], [1e-4, 1e-4]]],  # This has one std value larger than std_upper
            [[[1e-4, 1e-4], [1e-4, 1e-4]]],
        ])

        i_idx = torch.tensor([0])
        j_idx = torch.tensor([1])

        mask = gf.get_pairwise_valid_pixel_mask(
            image_stack,
            i_idx,
            j_idx,
            image_std_stack=std_stack,
            val_lower=val_lower,
            val_upper=val_upper,
            std_lower=std_lower,
            std_upper=std_upper
        )

        expected_mask = torch.tensor([
            [[[True, False], [True, True]]],
        ])

        assert torch.equal(mask, expected_mask)

    def test_get_pairwise_valid_pixel_mask_bad_threshold_args(self):

        image_stack, i_idx, j_idx = torch.ones((2, 2)), torch.tensor([0]), torch.tensor([0])
        val_lower, val_upper, std_lower, std_upper = 0.5, 0.1, 0.5, 0.1

        with pytest.raises(ValueError):
            gf.get_pairwise_valid_pixel_mask(image_stack, i_idx, j_idx, val_lower=val_lower, val_upper=val_upper)

        with pytest.raises(ValueError):
            gf.get_pairwise_valid_pixel_mask(image_stack, i_idx, j_idx, std_lower=std_lower, std_upper=std_upper)


class TestCvToTorch:

    def test_grayscale_input(self):
        # Grayscale image: shape (H, W)
        img = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        result = gf.cv_to_torch(img)

        assert result.shape == (1, 2, 2), "Grayscale image should be reshaped to (1, H, W)"
        assert torch.equal(result[0], img), "Grayscale content should remain unchanged"

    def test_color_input(self):
        # RGB image in OpenCV format: shape (H, W, 3) with BGR ordering
        img = torch.tensor([[[10, 20, 30], [40, 50, 60]],
                            [[70, 80, 90], [100, 110, 120]]], dtype=torch.uint8)
        # Convert to (C, H, W), BGR to RGB
        expected = img[:, :, [2, 1, 0]].permute(2, 0, 1)
        result = gf.cv_to_torch(img)

        assert result.shape == (3, 2, 2), "Color image should be reshaped to (3, H, W)"
        assert torch.equal(result, expected), "Color channels should be reordered from BGR to RGB"

    def test_invalid_input(self):
        # 1D tensor (invalid shape)
        with pytest.raises(ValueError, match="Unexpected image shape"):
            gf.cv_to_torch(torch.tensor([1, 2, 3]))

        # 3D tensor with wrong last dim
        bad_img = torch.zeros((10, 10, 4))
        with pytest.raises(ValueError, match="Unexpected image shape"):
            gf.cv_to_torch(bad_img)