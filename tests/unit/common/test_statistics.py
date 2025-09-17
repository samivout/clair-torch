import pytest

import random

from typeguard import TypeCheckError
import torch

from clair_torch.common.enums import VarianceMode
from clair_torch.common import statistics as st


class TestWBOMean:

    def test_WBOMean_init_success(self):

        dim = 2

        mean_manager = st.WBOMean(dim=dim)

        assert mean_manager.dim == (dim,)
        assert mean_manager.mean == 0.0
        assert mean_manager.sum_of_weights == 0.0

    def test_WBOMean_init_invalid_argument(self):

        dim = "bad"

        with pytest.raises(TypeCheckError):
            _ = st.WBOMean(dim=dim)

    def test_WBOMean_vs_simple_mean(self):

        dim = 0
        mean_manager = st.WBOMean(dim=dim)

        tensor_1 = torch.Tensor([1, 2, 3, 4, 5]).to(dtype=torch.float32)
        tensor_2 = torch.Tensor([6, 7, 8, 9, 10]).to(dtype=torch.float32)
        tensor_3 = torch.Tensor([11, 12, 13, 14, 15]).to(dtype=torch.float32)

        updates = [tensor_3, tensor_1, tensor_2]

        combined_tensor = torch.arange(start=1, end=16, step=1).to(dtype=torch.float32)

        expected_mean = combined_tensor.mean()
        for update in updates:
            mean_manager.update_values(update)

        batched_mean = mean_manager.mean

        assert torch.allclose(expected_mean, batched_mean, atol=1e-8)

    def test_WBOMean_vs_weighted_mean(self):

        dim = 0
        mean_manager = st.WBOMean(dim=dim)

        tensor_1 = torch.Tensor([1, 2, 3, 4, 5]).to(dtype=torch.float32)
        tensor_2 = torch.Tensor([6, 7, 8, 9, 10]).to(dtype=torch.float32)
        tensor_3 = torch.Tensor([11, 12, 13, 14, 15]).to(dtype=torch.float32)

        weights_1 = tensor_1 * 0.1
        weights_2 = tensor_2 * 0.1
        weights_3 = tensor_3 * 0.1

        updates = [(tensor_3, weights_3), (tensor_1, weights_1), (tensor_2, weights_2)]

        combined_tensor = torch.arange(start=1, end=16, step=1).to(dtype=torch.float32)
        combined_weights = combined_tensor * 0.1

        expected_mean = (torch.sum(combined_tensor * combined_weights, dim=dim, keepdim=True) /
                         torch.sum(combined_weights, dim=dim, keepdim=True))

        for update in updates:
            mean_manager.update_values(update[0], update[1])

        batched_mean = mean_manager.mean

        assert torch.allclose(expected_mean, batched_mean, atol=1e-8)


class TestWBOMeanVar:

    def test_WBOMeanVar_init_success(self):
        dim = 2

        mean_manager = st.WBOMeanVar(dim=dim)

        assert mean_manager.dim == (dim,)
        assert mean_manager.mean == 0.0
        assert mean_manager.sum_of_weights == 0.0

    def test_WBOMeanVar_init_fail(self):
        dim = "bad"

        with pytest.raises(TypeCheckError):
            _ = st.WBOMeanVar(dim=dim)

    @pytest.mark.parametrize("variance_mode",
                             [VarianceMode.POPULATION, VarianceMode.SAMPLE_FREQUENCY, VarianceMode.RELIABILITY_WEIGHTS])
    def test_WBOMeanVar_vs_simple_mean_and_variance(self, variance_mode):

        dim = 0
        mean_manager = st.WBOMeanVar(dim=dim, variance_mode=variance_mode)

        tensor_1 = torch.Tensor([1, 2, 3, 4, 5]).to(dtype=torch.float64)
        tensor_2 = torch.Tensor([6, 7, 8, 9, 10]).to(dtype=torch.float64)
        tensor_3 = torch.Tensor([11, 12, 13, 14, 15]).to(dtype=torch.float64)

        updates = [tensor_3, tensor_1, tensor_2]

        combined_tensor = torch.arange(start=1, end=16, step=1).to(dtype=torch.float64)
        number_of_items = len(combined_tensor)

        if variance_mode == VarianceMode.POPULATION:
            scale = 1 / number_of_items
        elif variance_mode == VarianceMode.SAMPLE_FREQUENCY:
            scale = 1 / (number_of_items - 1)
        elif variance_mode == VarianceMode.RELIABILITY_WEIGHTS:
            scale = 1 / (number_of_items - 1)
        else:
            raise ValueError(f"Unknown variance mode {variance_mode}")

        expected_mean = torch.mean(combined_tensor, dim=dim, keepdim=True)
        expected_variance = torch.sum((combined_tensor - expected_mean) ** 2, dim=dim, keepdim=True) * scale
        for update in updates:
            mean_manager.update_values(update)

        batched_mean = mean_manager.mean
        batched_var = mean_manager.variance()

        assert torch.allclose(expected_mean, batched_mean, atol=1e-8)
        assert torch.allclose(expected_variance, batched_var, atol=1e-8)

    @pytest.mark.parametrize("variance_mode", [VarianceMode.POPULATION, VarianceMode.SAMPLE_FREQUENCY, VarianceMode.RELIABILITY_WEIGHTS])
    def test_WBOMeanVar_vs_weighted_mean_and_variance_small_dataset(self, variance_mode):

        dim = 0
        mean_manager = st.WBOMeanVar(dim=dim, variance_mode=variance_mode)

        tensor_1 = torch.Tensor([1, 2, 3, 4, 5]).to(dtype=torch.float64)
        tensor_2 = torch.Tensor([6, 7, 8, 9, 10]).to(dtype=torch.float64)
        tensor_3 = torch.Tensor([11, 12, 13, 14, 15]).to(dtype=torch.float64)

        weights_1 = tensor_1 * 0.1
        weights_2 = tensor_2 * 0.1
        weights_3 = tensor_3 * 0.1

        updates = [(tensor_3, weights_3), (tensor_1, weights_1), (tensor_2, weights_2)]

        combined_tensor = torch.arange(start=1, end=16, step=1).to(dtype=torch.float64)
        combined_weights = combined_tensor * 0.1

        total_weights = torch.sum(combined_weights, dim=dim, keepdim=True)
        total_suqared_weights = torch.sum(combined_weights ** 2, dim=dim, keepdim=True)
        expected_mean = torch.sum(combined_tensor * combined_weights, dim=dim, keepdim=True) / total_weights

        if variance_mode == VarianceMode.RELIABILITY_WEIGHTS:
            scale = (total_weights / (total_weights ** 2 - total_suqared_weights))
        if variance_mode == VarianceMode.POPULATION:
            scale = 1 / total_weights
        if variance_mode == VarianceMode.SAMPLE_FREQUENCY:
            scale = 1 / (total_weights - 1)

        expected_variance = scale * torch.sum(combined_weights * (combined_tensor - expected_mean) ** 2, dim=dim, keepdim=True)

        for update in updates:
            mean_manager.update_values(update[0], update[1])

        batched_mean = mean_manager.mean
        batched_variance = mean_manager.variance()

        assert torch.allclose(expected_mean, batched_mean, atol=1e-8)
        assert torch.allclose(expected_variance, batched_variance, atol=1e-8)

    @pytest.mark.parametrize("variance_mode", [
        VarianceMode.POPULATION,
        VarianceMode.SAMPLE_FREQUENCY,
        VarianceMode.RELIABILITY_WEIGHTS
    ])
    def test_WBOMeanVar_vs_weighted_mean_and_variance_large_dataset(self, variance_mode):
        torch.manual_seed(42)
        random.seed(42)

        N = 10000
        values = torch.linspace(1, N, N, dtype=torch.float64)
        weights = torch.rand_like(values) + 0.5  # strictly positive

        # Direct computation
        total_w = torch.sum(weights)
        total_w_sq = torch.sum(weights ** 2)
        mean_direct = torch.sum(values * weights) / total_w

        if variance_mode == VarianceMode.POPULATION:
            scale = 1 / total_w
        elif variance_mode == VarianceMode.SAMPLE_FREQUENCY:
            scale = 1 / (total_w - 1)
        elif variance_mode == VarianceMode.RELIABILITY_WEIGHTS:
            scale = 1 / (total_w - total_w_sq / total_w)
        else:
            raise ValueError(f"Unknown variance mode {variance_mode}")

        var_direct = torch.sum(weights * (values - mean_direct) ** 2) * scale

        # Batched incremental computation
        mv = st.WBOMeanVar(dim=0, variance_mode=variance_mode)

        indices = list(range(len(values)))
        random.shuffle(indices)
        batch_size = 50
        for i in range(0, len(indices), batch_size):
            idx = indices[i:i + batch_size]
            mv.update_values(values[idx], weights[idx])

        batched_mean = mv.mean
        batched_var = mv.variance()

        # Tighter tolerances because large N reduces numerical error
        assert torch.allclose(mean_direct, batched_mean, atol=1e-10)
        assert torch.allclose(var_direct, batched_var, atol=1e-10)
