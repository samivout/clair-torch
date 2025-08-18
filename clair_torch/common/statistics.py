"""
Module for classes that can be used to easily compute mean and variance values of datasets in a batched online manner,
i.e. in one pass of the dataset.
"""
from typing import Optional
import torch

from clair_torch.common.enums import VarianceMode
from clair_torch.validation.type_checks import validate_all


class WBOMean(object):

    def __init__(self, dim: int | tuple[int, ...] = 0):
        """
        Weighted batched online mean (WBOMean). Allows the computation of a weighted mean value of a dataset in a single
        pass. Attributes are read-only properties and the user only needs to assign the dimension(s) over which the
        value is computed and supply the batches of new values and their associated weights.
        Args:
            dim: the dimension(s) along which to compute the values.
        """
        if isinstance(dim, int):
            dim = (dim,)
        elif isinstance(dim, tuple):
            validate_all(dim, int, raise_error=True, allow_none_iterable=False)
        else:
            raise TypeError(f"Expected dim as int or tuple of int, got {type(dim)}")

        self._dim = dim
        self._mean = 0.0
        self._sum_of_weights = 0.0

    @property
    def mean(self) -> float | torch.Tensor:
        return self._mean

    @property
    def sum_of_weights(self) -> float | torch.Tensor:
        return self._sum_of_weights

    @property
    def dim(self) -> int | tuple[int, ...]:
        return self._dim

    def internal_detach(self, *, in_place: bool = True):
        """
        Break the autograd graph attached to the internal state.
        Args:
            in_place: Whether to call detach so the tensor is modified in-place or to put into a new tensor.
        """
        if torch.is_tensor(self._mean):
            if in_place:
                self._mean.detach_()
            else:
                self._mean = self._mean.detach()

        if torch.is_tensor(self._sum_of_weights):
            if in_place:
                self._sum_of_weights.detach_()
            else:
                self._sum_of_weights = self._sum_of_weights.detach()

    def update_values(self, batch_values: torch.Tensor, batch_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The public method for updating the weighted mean value. Returns the new mean on update.
        Args:
            batch_values: the new batch of values used for updating the collective weighted mean.
            batch_weights: the new batch of weights associated with the given batch_values.

        Returns:
            The new mean value as a float or Tensor.
        """
        batch_size = int(torch.prod(torch.tensor([batch_values.shape[d] for d in self.dim])))

        if batch_weights is not None:
            total_batch_weights = torch.sum(batch_weights, dim=self.dim, keepdim=True)
            total_batch_mean = torch.sum(batch_weights * batch_values, dim=self.dim, keepdim=True) / (
                    total_batch_weights + 1e-6)
        else:
            total_batch_mean = torch.mean(batch_values, dim=self.dim, keepdim=True)
            total_batch_weights = torch.full_like(total_batch_mean, batch_size, dtype=batch_values.dtype)

        self._update_internal_values(total_batch_mean, total_batch_weights,
                                     None, None)
        return self.mean

    def _update_internal_values(self, total_batch_mean: torch.Tensor, total_batch_weights: torch.Tensor,
                                total_squared_batch_weights: torch.Tensor, total_batch_variance: torch.Tensor):
        """
        Implementation of the updating of the weighted mean according to DOI 10.1007/s00180-015-0637-z.
        Args:
            total_batch_mean: the mean of the new batch.
            total_batch_weights: the sum of the weights in the new batch.
            total_squared_batch_weights: unused dummy variable for inheriting class.
            total_batch_variance: unused dummy variable for inheriting class.
        Returns:

        """
        W_A = self.sum_of_weights
        W_B = total_batch_weights
        mean_A = self.mean
        mean_B = total_batch_mean
        W = W_A + W_B

        new_mean = mean_A + (W_B / W) * (mean_B - mean_A)

        self._mean = new_mean
        self._sum_of_weights = W


class WBOMeanVar(WBOMean):

    def __init__(self, dim: int | tuple[int, ...] = 0, variance_mode: VarianceMode = VarianceMode.RELIABILITY_WEIGHTS):
        """
        Weighted batched online mean and variance. Allows the computation of a weighted mean and variance of a dataset in a
        single pass. Attributes are read-only properties and the user only needs to assign the dimension(s) over which the
        value is computed and supply the batches of new values and their associated weights.
        Args:
            dim: the dimension(s) along which to compute the values.
            variance_mode: determines the normalization method to compute the variance.
        """
        super().__init__(dim=dim)

        self._variance_mode = variance_mode

        self._dispatch = {
            VarianceMode.POPULATION: self._variance_population,
            VarianceMode.RELIABILITY_WEIGHTS: self._variance_reliability_weights,
            VarianceMode.SAMPLE_FREQUENCY: self._variance_sample_frequency
        }

        if self._variance_mode not in self._dispatch:
            raise ValueError(f"Unknown variance mode {self._variance_mode}")

        self._m2 = 0.0
        self._sum_of_squared_weights = 0.0

    @property
    def sum_of_squared_weights(self) -> float | torch.Tensor:
        return self._sum_of_squared_weights

    @property
    def m2(self) -> float | torch.Tensor:
        return self._m2

    def variance(self) -> float | torch.Tensor:
        return self._dispatch[self._variance_mode]()

    def _variance_sample_frequency(self) -> float | torch.Tensor:
        return self._m2 * self._sample_frequency_scale(self._sum_of_weights)

    def _variance_reliability_weights(self) -> float | torch.Tensor:
        return self._m2 * self._reliability_weights_scale(self._sum_of_weights, self._sum_of_squared_weights)

    def _variance_population(self) -> float | torch.Tensor:
        return self._m2 * self._population_scale(self._sum_of_weights)

    @staticmethod
    def _sample_frequency_scale(sum_of_weights: float | torch.Tensor):
        return 1 / (sum_of_weights - 1)

    @staticmethod
    def _reliability_weights_scale(sum_of_weights: float | torch.Tensor, sum_of_squared_weights: float | torch.Tensor):
        return 1 / (sum_of_weights - sum_of_squared_weights / sum_of_weights)

    @staticmethod
    def _population_scale(sum_of_weights: float | torch.Tensor):
        return 1 / sum_of_weights

    def internal_detach(self, *, in_place: bool = True) -> None:
        """
        Break the autograd graph attached to the internal state.
        Args:
            in_place: Whether to call detach so the tensor is modified in-place or to put into a new tensor.
        """
        if torch.is_tensor(self._mean):
            if in_place:
                self._mean.detach_()
            else:
                self._mean = self._mean.detach()

        if torch.is_tensor(self._m2):
            if in_place:
                self._m2.detach_()
            else:
                self._m2 = self._m2.detach()

        if torch.is_tensor(self._sum_of_weights):
            if in_place:
                self._sum_of_weights.detach_()
            else:
                self._sum_of_weights = self._sum_of_weights.detach()

        if torch.is_tensor(self._sum_of_squared_weights):
            if in_place:
                self._sum_of_squared_weights.detach_()
            else:
                self._sum_of_squared_weights = self._sum_of_squared_weights.detach_()

    def update_values(self, batch_values: torch.Tensor, batch_weights: Optional[torch.Tensor] = None) -> \
            tuple[float | torch.Tensor, float | torch.Tensor]:
        """
        The public method for updating the weighted mean value. Returns the new mean on update.
        Args:
            batch_values: the new batch of values used for updating the collective weighted mean.
            batch_weights: the new batch of weights associated with the given batch_values.

        Returns:
            The new mean value and variance values as tuple of floats or tensors.
        """
        batch_size = int(torch.prod(torch.tensor([batch_values.shape[d] for d in self.dim])))

        if batch_weights is not None:
            total_batch_weights = torch.sum(batch_weights, dim=self.dim, keepdim=True)
            total_squared_batch_weights = torch.sum(batch_weights ** 2, dim=self.dim, keepdim=True)
            total_batch_mean = torch.sum(batch_weights * batch_values, dim=self.dim, keepdim=True) / (
                    total_batch_weights + 1e-6)
            m2 = torch.sum(batch_weights * (batch_values - total_batch_mean) ** 2, dim=self.dim, keepdim=True)
        else:
            total_batch_mean = torch.mean(batch_values, dim=self.dim, keepdim=True)
            total_batch_weights = torch.full_like(total_batch_mean, batch_size, dtype=batch_values.dtype)
            total_squared_batch_weights = total_batch_weights
            m2 = torch.sum((batch_values - total_batch_mean) ** 2, dim=self.dim, keepdim=True)

        total_batch_variance = m2

        self._update_internal_values(total_batch_mean, total_batch_weights,
                                     total_squared_batch_weights, total_batch_variance)
        return self.mean, self.m2

    def _update_internal_values(self, total_batch_mean: torch.Tensor, total_batch_weights: torch.Tensor,
                                total_squared_batch_weights: torch.Tensor, total_batch_variance: torch.Tensor) -> None:
        """
        Implementation of the updating of the weighted mean according to DOI 10.1007/s00180-015-0637-z.
        Args:
            total_batch_mean: the mean of the new batch.
            total_batch_weights: the sum of the weights in the new batch.

        Returns:
            None
        """
        W_A = self.sum_of_weights
        W_B = total_batch_weights
        M_A = self.m2
        M_B = total_batch_variance
        mean_A = self.mean
        mean_B = total_batch_mean
        W = W_A + W_B

        M_AB = M_A + M_B + (W_A * W_B / W) * (mean_B - mean_A) ** 2
        new_mean = mean_A + (W_B / W) * (mean_B - mean_A)

        self._mean = new_mean
        self._m2 = M_AB
        self._sum_of_weights = W
        self._sum_of_squared_weights += total_squared_batch_weights
