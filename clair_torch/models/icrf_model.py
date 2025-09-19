"""
Module for the concrete implementations of the ICRFModelBase class
"""
from typing import Optional

import torch
import torch.nn as nn
from typeguard import typechecked

from clair_torch.models.base import ICRFModelBase
from clair_torch.common.enums import InterpMode


class ICRFModelPCA(ICRFModelBase):
    """
    ICRF model class that utilizes a set of principal components for the optimization process. For each principal
    component a single scalar optimization parameter is utilized, e.g. for 5 components there is a total of 5 parameters
    to optimize in the model. The shape of the given principal components is used to determine the number of datapoints,
    number of components and number of channels. In addition to these an exponent for the linear range [0, 1] is used
    as a parameter for each channel.

    Attributes:
    -----------
    Inherits attributes from ICRFModelBase

    p: nn.ParameterList
        a list of nn parameters representing the exponent of the base curve. One element for each channel.
    coefficients: nn.ParameterList
        a list of nn parameters representing the PCA parameters, for each channel a number equal to the number of
            components.
    """
    @typechecked
    def __init__(self, pca_basis: torch.Tensor, interpolation_mode: InterpMode = InterpMode.LINEAR,
                 initial_power: float = 2.5, icrf: Optional[torch.Tensor] = None) -> None:
        """
        Initializes a ICRFModelPCA instance. The pca_basis is used to determine the shape of the actual ICRF.
        Args:
            pca_basis: a torch.Tensor representing the principal components. The shape is expected to be
                (n_points, num_components, channels).
            interpolation_mode: a InterpMode determining how the ICRF is used in a forward call.
            initial_power: a guess at the initial form of the ICRF curve, represented by raising the linear range [0, 1]
                to this power.
            icrf: an optional initial form of the ICRF curves, overrides n_points and channels.
        """
        n_points, num_components, channels = pca_basis.shape

        super().__init__(n_points, channels, interpolation_mode, initial_power, icrf)

        # Base curve power (learnable scalar), one parameter per channel.
        self.p = nn.ParameterList([nn.Parameter(torch.tensor(2.0)) for _ in range(channels)])

        # Model weights, num_components number per channel.
        self.coefficients = nn.ParameterList([nn.Parameter(torch.zeros(num_components)) for _ in range(channels)])

        self.register_buffer("pca_basis", pca_basis)
        self.register_buffer("x_values", torch.linspace(0, 1, n_points))  # (L,)

    def channel_params(self, c: int) -> list[nn.Parameter]:
        """
        Main method for accessing the optimization parameters of the model.
        Args:
            c: channel index.

        Returns:
            A list of nn.Parameters.
        """
        return [self.p[c], self.coefficients[c]]

    def update_icrf(self):
        """
        Builds the full ICRF curve (n_points, channels) from base + PCA.
        """
        p_tensor = torch.stack(list(self.p))
        coefficient_tensor = torch.stack(list(self.coefficients))

        # x_values: (L,)
        x_safe = self.x_values.clamp(min=1e-6).unsqueeze(1)  # (L, 1)

        # Compute per-channel base curve: (L, C)
        base_curve = x_safe.pow(p_tensor.unsqueeze(0))  # broadcasted over L

        # PCA component contribution: (L, C)
        pca_curve = (self.pca_basis * coefficient_tensor.T.unsqueeze(0)).sum(dim=1)

        # Combine base and PCA to update self.icrf
        self._icrf = base_curve + pca_curve  # shape: (L, C)


class ICRFModelDirect(ICRFModelBase):
    """
    An ICRF model class that utilizes the datapoints directly as optimization parameters. Total number of parameters is
    therefore n_points * channels.

    Attributes:
    ----------
    Inherits attributes from ICRFModelBase.

    direct_params: nn.ParameterList
        a list of nn parameters, each parameter corresponds to an actual datapoint in the ICRF curve.
    """
    @typechecked
    def __init__(self, n_points: Optional[int] = 256, channels: Optional[int] = 3,
                 interpolation_mode: InterpMode = InterpMode.LINEAR, initial_power: float = 2.5,
                 icrf: Optional[torch.Tensor] = None):

        super().__init__(n_points, channels, interpolation_mode, initial_power, icrf)

        self.direct_params = nn.ParameterList([
            nn.Parameter(torch.linspace(0, 1, n_points) ** initial_power) for _ in range(channels)
        ])

    def channel_params(self, c: int):
        """
        Main method for accessing the optimization parameters of the model.
        Args:
            c: channel index.

        Returns:
            A list of nn.Parameters.
        """
        return [self.direct_params[c]]

    def update_icrf(self):
        """
        Directly stack the per-channel parameters to form the ICRF.
        """
        self._icrf = torch.stack([p for p in self.direct_params], dim=0)  # shape: (L, C)


if __name__ == "__main__":
    pass
