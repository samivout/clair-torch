"""
Module for the base model classes.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from typeguard import typechecked

from clair_torch.common.enums import InterpMode
from clair_torch.visualization.plotting import plot_data_and_diff


class ICRFModelBase(nn.Module, ABC):
    """
    Base class for the ICRF model classes. Implements common functionality and acts as a guideline for implementing the
    model interface.

    Attributes
    ----------
    _n_points: int
        how many datapoints the range [0, 1] is split into in modelling the ICRF.
    _channels: int
        how many color channels are managed by the model. One ICRF curve for each channel.
    interpolation_mode: InterpMode
        enum determining how the forward call of the model is handled. See InterpMode doc for more.
    _initial_power: float
        a guess at the initial form of the ICRF curve, represented by raising the linear range [0, 1] to this power.
    _fig: Figure
        a matplotlib Figure used for visualizing the model.
    _axs: List[Axes]
        a list of matplotlib axes used for model visualization.
    _lines_curve: List[Line2D]
        a list of the matplotlib Line2D objects for the plotted curves used in model visualization.
    _lines_deriv: List[Line2D]
        a list of the matplotlib Line2D objects for the plotted curves' derivatives used in model visualization.
    """
    @typechecked
    def __init__(self, n_points: Optional[int] = 256, channels: Optional[int] = 3,
                 interpolation_mode: InterpMode = InterpMode.LINEAR, initial_power: float = 2.5,
                 icrf: Optional[torch.Tensor] = None):
        """
        Initializes the ICRF model instance with the given parameters. The icrf argument overrides n_points and channels
        by its shape if given.
        Args:
            n_points: how many datapoints the range [0, 1] is split into in modelling the ICRF.
            channels: how many datapoints the range [0, 1] is split into in modelling the ICRF.
            interpolation_mode: enum determining how the forward call of the model is handled. See InterpMode doc for
                more.
            initial_power: a guess at the initial form of the ICRF curve, represented by raising the linear range [0, 1]
                to this power.
            icrf: an optional initial form of the ICRF curves, overrides n_points and channels.
        """
        super().__init__()

        # If an ICRF curve is given, its shape will override the given n_points and channels parameters.
        if icrf is not None:
            channels, n_points = icrf.shape

        self._channels = channels
        self._initial_power = initial_power
        self._n_points = n_points
        self.register_buffer("_x_axis_datapoints", torch.linspace(0, 1, n_points))

        if icrf is None:
            icrf = self._initialize_default_icrf()
        self.register_buffer("_icrf", icrf)

        self.interpolation_mode = interpolation_mode

        self._dispatch = {
            InterpMode.LOOKUP: self._forward_lookup,
            InterpMode.LINEAR: self._forward_linear,
            InterpMode.CATMULL: self._forward_catmull,
        }

        if self.interpolation_mode not in self._dispatch:
            raise ValueError(f'Unknown interpolation mode {self.interpolation_mode}')

        self._fig = None
        self._axs = None
        self._lines_curve = None
        self._lines_deriv = None

    @property
    def icrf(self):
        return self._icrf

    @property
    def channels(self):
        return self._channels

    @property
    def n_points(self):
        return self._n_points

    @property
    def initial_power(self):
        return self._initial_power

    @property
    def x_axis_datapoints(self):
        return self._x_axis_datapoints

    @abstractmethod
    def channel_params(self, c: int) -> list[nn.Parameter]:
        """
        Method for getting the model parameters for the channel of the given index. Subclasses implement the logic
        based on their model parameters. This should be the main method of accessing the optimization parameters for
        feeding them to a torch.Optimizer.
        Args:
            c: channel index.

        Returns:
            list of nn.Parameters.
        """

    @abstractmethod
    def update_icrf(self) -> None:
        """
        Method for constructing a new ICRF curve from the model parameters and updating the curve to the
        self._icrf attribute. Subclasses implement the logic based their model parameters.
        """

    def _initialize_default_icrf(self) -> torch.Tensor:
        """
        Initializes a default ICRF curve if None is given, based on the number of datapoints, channels and the value
        of the initial power.
        """
        return torch.transpose(torch.linspace(0, 1, self.n_points).unsqueeze(1).repeat(1, self.channels) ** self.initial_power, 0, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._dispatch[self.interpolation_mode](image)

    def _forward_lookup(self, image: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbour table lookup (no gradients).
        image: (N, C, H, W) with values in [0, 1]
        self.icrf: (L, C) first dim = curve sample, second dim = channel
        """
        N, C, H, W = image.shape
        L = self.icrf.shape[1]

        # integer sample index per pixel
        idx = (image * (L - 1)).round().clamp(0, L - 1).long()  # (N, C, H, W)

        # matching channel index tensor
        chan = (
            torch.arange(C, device=image.device)
            .view(1, C, 1, 1)
            .expand(N, C, H, W)  # same shape as idx
        )

        # advanced indexing: returns (N, C, H, W)
        return self.icrf[chan, idx]

    def _forward_linear(self, image: torch.Tensor) -> torch.Tensor:
        # image : (N, C, H, W) in [0, 1]
        N, C, H, W = image.shape
        L = self.icrf.size(1)

        # scale pixel values to LUT index range [0, L-1]
        x = (image * (L - 1)).clamp_(0, L - 1)

        x0 = x.floor().long()  # lower index
        x1 = (x0 + 1).clamp_(0, L - 1)  # upper index
        w = (x - x0.float())  # weight  (N,C,H,W)

        # helper that gathers LUT values for arbitrary index tensor
        def gather(ix):
            flat_ix = ix.reshape(-1)
            # Build channel indices with same layout as ix
            chan = torch.arange(C, device=ix.device).view(1, C, 1, 1)
            chan = chan.expand(N, C, H, W).reshape(-1)
            return self.icrf[chan, flat_ix].reshape(N, C, H, W)

        g0 = gather(x0)
        g1 = gather(x1)

        # linear interpolation
        return g0 * (1.0 - w) + g1 * w

    def _forward_catmull(self, image: torch.Tensor) -> torch.Tensor:
        # Assume image shape: (N, C, H, W), values in [0, 1]
        N, C, H, W = image.shape
        L = self.icrf.shape[1]  # number of ICRF samples

        # Scale input to [0, L - 1]
        x = (image * (L - 1)).clamp(0, L - 1)

        # Compute indices for Catmull-Rom interpolation
        x0 = x.floor().long()
        x_indices = [  # Collect 4 neighboring indices: x-1, x0, x0+1, x0+2
            (x0 - 1).clamp(0, L - 1),
            x0.clamp(0, L - 1),
            (x0 + 1).clamp(0, L - 1),
            (x0 + 2).clamp(0, L - 1),
        ]

        # Compute fractional part
        t = (x - x0.float()).clamp(0, 1)

        # Catmull-Rom basis weights
        t2 = t * t
        t3 = t2 * t

        w0 = -0.5 * t3 + t2 - 0.5 * t
        w1 = 1.5 * t3 - 2.5 * t2 + 1.0
        w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        w3 = 0.5 * t3 - 0.5 * t2

        weights = [w0, w1, w2, w3]  # each (N, C, H, W)

        # Broadcast gather: returns (N, C, H, W) for each index tensor
        def gather(ix):
            flat_ix = ix.reshape(-1)
            # Build channel indices with same layout as ix
            chan = torch.arange(C, device=ix.device).view(1, C, 1, 1)
            chan = chan.expand(N, C, H, W).reshape(-1)
            return self.icrf[chan, flat_ix].reshape(N, C, H, W)

        g = [gather(ix) for ix in x_indices]

        # Final smooth interpolated result
        result = torch.stack([w * gi for w, gi in zip(weights, g)], dim=0).sum(dim=0)

        return result

    def _prepare_icrf_plot_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal utility function to prepare the model data for plotting.
        Returns:
            Tuple of NumPy arrays representing [x_values, y_values, dy/dx_values].
        """
        icrf_cpu = self.icrf.detach().cpu().numpy()
        x = np.linspace(0, 1, self.n_points)
        dy = np.diff(icrf_cpu, axis=1)
        dx = x[1] - x[0]
        dydx = dy / dx

        return x, icrf_cpu, dydx

    def plot_icrf(self) -> None:
        """
        Model utility function for live-plotting. The model class manages the state of the plot, while utilizing the
        general plotting function of the plotting module.
        """
        x, icrf_cpu, dydx = self._prepare_icrf_plot_data()

        if self._fig is None:
            plt.ion()  # Enable interactive mode for live plotting.
            self._fig, self._axs, self._lines_curve, self._lines_deriv = plot_data_and_diff(x, icrf_cpu, dydx)
        else:
            for c in range(self.channels):
                self._lines_curve[c].set_ydata(icrf_cpu[c, :])
                self._lines_deriv[c].set_ydata(dydx[c, :])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)
