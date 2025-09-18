from typing import Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def plot_linearity_loss(exposure_ratios: torch.Tensor, spatial_means: torch.Tensor,
                        spatial_stds: Optional[torch.Tensor] = None,
                        spatial_errors: Optional[torch.Tensor] = None) -> Figure:
    """
    Function for creating the linearity loss plots used in linearity measurements. The plot is based on the exposure
    time ratios and the spatial means, standard deviations and uncertainties of the linearity losses of pairs of images
    captured at different exposure times, but under stationary conditions.
    Args:
        exposure_ratios: tensor of the used exposure time ratios.
        spatial_means: tensor of the spatial means of linearity loss.
        spatial_stds: tensor of the spatial standard deviations of linearity loss.
        spatial_errors: tensor of the uncertainties of the spatial means of linearity loss.

    Returns:
        A matplotlib Figure.
    """
    exposure_ratios_np = exposure_ratios.detach().cpu().numpy()
    spatial_means_np = spatial_means.detach().cpu().numpy()
    if spatial_stds is not None:
        spatial_stds_np = spatial_stds.detach().cpu().numpy()
    else:
        spatial_stds_np = None
    if spatial_errors is not None:
        spatial_errors_np = spatial_errors.detach().cpu().numpy()
    else:
        spatial_errors_np = None

    symbol = "R"
    ylabel = "Relative disparity"

    channels = np.shape(spatial_means_np)[-1]   # Last channel assumed to be the channels dimension.
    fig, axes = plt.subplots(1, channels, figsize=(20, 5))
    colors = ['r', 'g', 'b']

    for c, ax in enumerate(axes):

        color = colors[c]

        y = spatial_means_np[:, c]
        ax.scatter(exposure_ratios_np, y, c='black', marker='x')

        if spatial_stds_np is not None:
            y_std = spatial_stds_np[:, c]
            ax.errorbar(exposure_ratios_np, y, yerr=(y_std / 5), elinewidth=1,
                        c=color, marker=None, linestyle='none', markersize=3, alpha=0.5,
                        label=fr'$\sigma_{{{color}, {symbol}}}$')

        if spatial_errors_np is not None:
            y_err = spatial_errors_np[:, c]
            ax.errorbar(exposure_ratios_np, y, yerr=y_err, elinewidth=1,
                        c='black', marker=None, linestyle='none', markersize=3, alpha=0.5,
                        label=fr'$\sigma_{{{color}, {symbol}}}$')

        ax.legend(loc='best')

    axes[0].set(ylabel=ylabel)
    axes[1].set(xlabel=r'Exposure time ratio $t_s/t_l$')
    axes[0].yaxis.label.set_size(16)
    axes[1].xaxis.label.set_size(16)

    return fig


def plot_data_and_diff(x_datapoints: np.ndarray, datapoints: np.ndarray, datapoints_diff: np.ndarray) \
        -> tuple[Figure, list[Axes], list[Line2D], list[Line2D]]:
    """
    Function for creating the figure and its contents for two sets of datapoints over a common x-axis in two
    subplots.
    Args:
        x_datapoints: the x datapoints.
        datapoints: the first set of y datapoints.
        datapoints_diff: the second set of y datapoints.

    Returns:
        A plt Figure, list of Axes and two lists of Line2D. Number of items in lists is equivalent to number of
        channels in the given data.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    lines_curve, lines_deriv = [], []

    for c in range(datapoints.shape[0]):
        (line,) = axs[0].plot(x_datapoints, datapoints[c, :], label=f"Channel {c}")
        (dline,) = axs[1].plot(x_datapoints[:-1], datapoints_diff[c, :], label=f"Channel {c}")
        lines_curve.append(line)
        lines_deriv.append(dline)

    axs[0].set_title("ICRF Transfer Functions")
    axs[1].set_title("ICRF Derivatives")
    for ax in axs:
        ax.grid(True)
        ax.legend()
    plt.tight_layout()

    return fig, axs, lines_curve, lines_deriv


def plot_two_ICRF_and_calculate_RMSE(file_path1: Path, file_path2: Path, output_path: Path):

    name = 'ICRF_RMSE.png'

    ICRF1 = np.loadtxt(file_path1)  # shape: (L, C)
    ICRF2 = np.loadtxt(file_path2)  # shape: (L, C)

    # Check shapes
    assert ICRF1.shape == ICRF2.shape, "ICRF arrays must have the same shape"
    L, C = ICRF1.shape

    RMSE = np.sqrt(np.mean((ICRF1 - ICRF2) ** 2))
    x_range = np.linspace(0, 1, L)

    fig, axs = plt.subplots(1, C, figsize=(5 * C, 4), squeeze=False)
    fig.suptitle(f'ICRF Comparison – RMSE: {RMSE:.4f}', fontsize=14)

    for c in range(C):
        ax = axs[0, c]
        ax.plot(x_range, ICRF1[:, c], label='ICRF 1', color='red')
        ax.plot(x_range, ICRF2[:, c], label='ICRF 2', color='blue')
        ax.set_title(f'Channel {c}')
        ax.set_xlabel('Normalized Brightness')
        ax.set_ylabel('Normalized Irradiance')
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_path.joinpath(name), dpi=300)
    plt.clf()

    return


def update_loss_plot(epoch: int, losses_per_channel: np.ndarray) -> None:
    """
    Function for initializing and also updating a loss plot for training.
    Args:
        epoch: the epoch of the new loss data.
        losses_per_channel: the losses of each channel as a NumPy array.
    """
    losses_per_channel = np.asarray(losses_per_channel, dtype=float)

    if not hasattr(update_loss_plot, "_state"):

        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Channel‑wise Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        channel_plots = []
        for c in range(len(losses_per_channel)):
            (line,) = ax.plot([epoch], [losses_per_channel[c]],
                              label=f"ch {c}")
            channel_plots.append(line)

        ax.legend()
        plt.tight_layout()

        update_loss_plot._state = {
            "fig": fig,
            "ax": ax,
            "channel_plots": channel_plots,
            "xs": [],
            "ys": [[] for _ in channel_plots]
        }

    st = update_loss_plot._state
    channel_plots = st["channel_plots"]

    st["xs"].append(epoch)
    for c, val in enumerate(losses_per_channel):
        st["ys"][c].append(val)
        channel_plots[c].set_data(st["xs"], st["ys"][c])

    st["ax"].relim()
    st["ax"].autoscale_view()

    st["fig"].canvas.draw_idle()
    st["fig"].canvas.flush_events()

    return


if __name__ == "__main__":
    pass
