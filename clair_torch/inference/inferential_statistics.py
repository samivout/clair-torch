"""
Module for computing statistics of datasets. Currently only contains the function for computing the mean and standard
deviation of all frames in a set of video files.
"""
from typing import Optional
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from clair_torch.common.enums import VarianceMode
from clair_torch.common.statistics import WBOMeanVar
from clair_torch.models.base import ICRFModelBase


@typechecked
def compute_video_mean_and_std(dataloader: DataLoader, device: str | torch.device,
                               icrf_model: Optional[ICRFModelBase] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function for computing the mean and standard deviation of the frames in a given dataset of video files. All frames
    in all the videos are treated as belonging in the same dataset.
    Args:
        dataloader: a DataLoader containing a VideoIterableDataset, representing the dataset.
        device: torch device to run the computation on.
        icrf_model: an ICRF model to optionally linearize the pixel values before computing the mean and std.

    Returns:
        The mean and standard deviations of the (possibly linearized) video frames in the dataset.
    """
    total_iterations = len(dataloader)

    mean_handler = WBOMeanVar(dim=0, variance_mode=VarianceMode.SAMPLE_FREQUENCY)
    number_of_frames = 0

    with torch.inference_mode():
        for idx_batch, val_batch, std_batch, meta_batch in tqdm(dataloader, desc="Number of batches processed",
                                                                total=total_iterations):

            frames = val_batch.to(device=device)
            number_of_frames += frames.shape[0]

            if icrf_model:
                frames = icrf_model(frames)

            mean_handler.update_values(frames, None)

    return mean_handler.mean.squeeze(), torch.sqrt(mean_handler.variance().squeeze()) / math.sqrt(number_of_frames)
