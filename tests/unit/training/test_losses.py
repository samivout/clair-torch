import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st

import torch

from clair_torch.training import losses as ls


class TestGaussianValueWeights:

    def test_gaussian_value_weights_zero_scale(self):
        image = 2 * torch.ones((3, 3), dtype=torch.float64)
        scale = 0.0

        weights = ls.gaussian_value_weights(image, scale)
        expected = torch.ones((3, 3), dtype=torch.float64)

        assert torch.allclose(weights, expected)

    def test_gaussian_value_weights_max_location_and_value(self):
        image = torch.tensor([0.0, 0.4, 0.5, 0.6, 0.1], dtype=torch.float64)

        weights = ls.gaussian_value_weights(image)

        assert weights[2] == 1.0
        assert weights.argmax() == 2

    @given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=100))
    def test_gaussian_value_weights_symmetry(self, values):
        image = torch.tensor(values, dtype=torch.float64)
        weights_a = ls.gaussian_value_weights(image)
        weights_b = ls.gaussian_value_weights(1.0 - image)

        assert torch.allclose(weights_a, weights_b, atol=1e-12)

    @given(st.lists(st.floats(min_value=0, max_value=1), min_size=1))
    def test_gaussian_value_weights_bounds(self, values):
        image = torch.tensor(values, dtype=torch.float64)
        weights = ls.gaussian_value_weights(image, scale=10.0)

        assert torch.all(weights > 0)
        assert torch.all(weights <= 1.0)

    def test_gaussian_value_weights_scale_monotonicity(self):
        image = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float64)
        w_small = ls.gaussian_value_weights(image, scale=5.0)
        w_large = ls.gaussian_value_weights(image, scale=50.0)
        mask = image != 0.5  # exclude the peak

        assert torch.all(w_large[mask] < w_small[mask])

    def test_gaussian_value_weights_grad_support(self):
        image = torch.linspace(0, 1, steps=5, dtype=torch.float64, requires_grad=True)
        weights = ls.gaussian_value_weights(image)

        assert weights.requires_grad
        weights.sum().backward()

        assert image.grad is not None


class TestCombinedGaussianPairWeights:

    def test_combined_gaussian_pair_weights(self):
        N, C, H, W = 3, 3, 5, 5

        image_stack = torch.ones((N, C, H, W), dtype=torch.float64)
        i_idx, j_idx = torch.triu_indices(N, N, offset=1)
        return_value = torch.ones((2, 3, 5, 5), dtype=torch.float64)

        with patch("clair_torch.training.losses.gaussian_value_weights", return_value=return_value):
            combined_weights = ls.combined_gaussian_pair_weights(image_stack, i_idx, j_idx)

        assert combined_weights.shape == (2, 3, 5, 5)
        assert torch.allclose(combined_weights, 2 * return_value)

    def test_combined_gaussian_pair_weights_grad_support(self):
        N, C, H, W = 3, 3, 5, 5

        image_stack = torch.ones((N, C, H, W), dtype=torch.float64, requires_grad=True)
        i_idx, j_idx = torch.triu_indices(N, N, offset=1)

        combined_weights = ls.combined_gaussian_pair_weights(image_stack, i_idx, j_idx)

        assert combined_weights.requires_grad
        combined_weights.sum().backward()

        assert image_stack.grad is not None


class TestComputeEndpointPenalty:

    def test_endpoint_penalty_single_channel(self):
        curve = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float64)
        expected_penalty = torch.tensor([2 * 0.5 ** 2], dtype=torch.float64)

        penalty = ls.compute_endpoint_penalty(curve, per_channel=True)

        assert torch.allclose(expected_penalty, penalty)

    @pytest.mark.parametrize("per_channel, expected", [
        (True, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)),
        (False, torch.tensor([1.5], dtype=torch.float64))
    ])
    def test_endpoint_penalty_multiple_channels(self, per_channel, expected):
        curve = torch.tensor([[0.5, 0.5, 0.5],
                              [0.0, 0.0, 0.0],
                              [0.5, 0.5, 0.5]], dtype=torch.float64)

        penalty = ls.compute_endpoint_penalty(curve, per_channel=per_channel)

        assert torch.allclose(expected, penalty)

    def test_endpoint_penalty_grad_support(self):
        curve = torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float64, requires_grad=True)

        penalty = ls.compute_endpoint_penalty(curve, per_channel=True)
        penalty.sum().backward()

        assert curve.grad is not None


class TestComputeRangePenalty:

    def test_compute_range_penalty_single_channel(self):

        curve = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64)
        expected_penalty = torch.tensor([4], dtype=torch.float64)

        penalty = ls.compute_range_penalty(curve, per_channel=True)

        assert torch.allclose(expected_penalty, penalty)

    @pytest.mark.parametrize("per_channel, expected", [
        (True, torch.tensor([4, 4, 4], dtype=torch.float64)),
        (False, torch.tensor([12], dtype=torch.float64))
    ])
    def test_compute_range_penalty_multiple_channels(self, per_channel, expected):

        curve = torch.tensor([[-2, -1, 0, 1, 2],
                              [-2, -1, 0, 1, 2],
                              [-2, -1, 0, 1, 2]], dtype=torch.float64).transpose(dim0=0, dim1=1)

        penalty = ls.compute_range_penalty(curve, per_channel=per_channel)

        assert torch.allclose(expected, penalty)

    def test_compute_range_penalty_grad_support(self):

        curve = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64, requires_grad=True)

        penalty = ls.compute_range_penalty(curve, per_channel=True)
        penalty.sum().backward()

        assert curve.grad is not None


class TestComputeSmoothnessPenalty:

    def test_compute_smoothness_penalty_single_channel(self):

        # For a quadratic curve the second difference is constant and easy to verify manually:
        # second_diff[i] = curve[i] - 2*curve[i+1] + curve[i+2]
        # For i=0: 0 - 2*1 + 4 = 2
        # For i=1: 1 - 2*4 + 9 = 2
        # For i=2: 4 - 2*9 + 16 = 2
        # So second_diff = [2, 2, 2]
        # penalty = sum( second_diff^2 ) = 3 * (2^2) = 12
        x = torch.arange(5, dtype=torch.float64).unsqueeze(1)
        curve = x.pow(2)

        expected_penalty = torch.tensor(12.0, dtype=torch.float64)

        penalty = ls.compute_smoothness_penalty(curve, per_channel=False)

        assert torch.allclose(penalty, expected_penalty)

    @pytest.mark.parametrize("per_channel, expected", [
            (True, torch.tensor([12.0, 12.0, 12.0], dtype=torch.float64)),
            (False, torch.tensor(36.0, dtype=torch.float64))
    ])
    def test_compute_smoothness_penalty_multiple_channels(self, per_channel, expected):
        x = torch.arange(5, dtype=torch.float64).unsqueeze(1)
        single_channel_curve = x.pow(2)

        curve = single_channel_curve.repeat(1, 3)

        penalty = ls.compute_smoothness_penalty(curve, per_channel=per_channel)

        assert torch.allclose(penalty, expected)

    def test_compute_smoothness_penalty_grad_support(self):

        x = torch.arange(5, dtype=torch.float64).unsqueeze(1)
        curve = x.pow(2)
        curve = curve.requires_grad_(True)

        penalty = ls.compute_smoothness_penalty(curve, per_channel=False)
        penalty.sum().backward()

        assert curve.grad is not None


class TestComputeMonotonicityPenalty:

    def test_compute_monotonicity_penalty(self):
        # Single-channel curve: strictly increasing, no penalty
        curve = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
        assert torch.allclose(ls.compute_monotonicity_penalty(curve), torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(ls.compute_monotonicity_penalty(curve, per_channel=True), torch.tensor([0.0], dtype=torch.float64))

        # Non-monotone: last step decreases by 2
        curve = torch.tensor([[0.0], [1.0], [3.0], [1.0]], dtype=torch.float64)
        assert torch.allclose(ls.compute_monotonicity_penalty(curve), torch.tensor(4.0, dtype=torch.float64))
        assert torch.allclose(ls.compute_monotonicity_penalty(curve, squared=False), torch.tensor(2.0, dtype=torch.float64))

