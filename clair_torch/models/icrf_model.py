from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn

from clair_torch.models.base import ICRFModelBase
from clair_torch.common.data_io import load_icrf_txt
from clair_torch.common.enums import InterpMode


class ICRFModelPCA(ICRFModelBase):

    def __init__(self, pca_basis: torch.Tensor, interpolation_mode: InterpMode = InterpMode.LINEAR,
                 initial_power: float = 2.5, icrf: Optional[torch.Tensor] = None) -> None:

        n_points, num_components, channels = pca_basis.shape

        super().__init__(n_points, channels, interpolation_mode, initial_power, icrf)

        # Base curve power (learnable scalar), one parameter per channel.
        self.p = nn.ParameterList([nn.Parameter(torch.tensor(2.0)) for _ in range(channels)])

        # Model weights, num_components number per channel.
        self.coefficients = nn.ParameterList([nn.Parameter(torch.zeros(num_components)) for _ in range(channels)])

        self.register_buffer("pca_basis", pca_basis)
        self.register_buffer("x_values", torch.linspace(0, 1, n_points))  # (L,)

        self._fig = None

    def channel_params(self, c: int) -> list[nn.Parameter]:
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
    def __init__(self, n_points: Optional[int] = 256, channels: Optional[int] = 3,
                 interpolation_mode: InterpMode = InterpMode.LINEAR, initial_power: float = 2.5,
                 icrf: Optional[torch.Tensor] = None):

        super().__init__(n_points, channels, interpolation_mode, initial_power, icrf)

        # Override PCA parameters with direct per-channel parameters
        self.direct_params = nn.ParameterList([
            nn.Parameter(torch.linspace(0, 1, n_points) ** initial_power) for _ in range(channels)
        ])

    def channel_params(self, c: int):
        return [self.direct_params[c]]

    def update_icrf(self):
        """
        Directly stack the per-channel parameters to form the ICRF.
        """
        self._icrf = torch.stack([p for p in self.direct_params], dim=1)  # shape: (L, C)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    icrf = load_icrf_txt(Path(r"E:\Project\camera_linearity\data\ICRF_calibrated.txt"))
    icrf = icrf.to(device)
    model = ICRFModelPCA(icrf=icrf).to(device)
    model.plot_icrf()
