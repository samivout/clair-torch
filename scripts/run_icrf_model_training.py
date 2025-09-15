"""
A ready-to-run script file for training an ICRF model. Reads the parameters from the config file pointed by be the
CLI argument.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clair_torch.common.parameters import Parameters
from clair_torch.common.general_functions import cli_parse_args_from_config
from clair_torch.common.file_settings import file_settings_constructor
from clair_torch.common.data_io import save_icrf_txt
from clair_torch.metadata.imaging_metadata import ImagingMetadata
from clair_torch.datasets.image_dataset import ImageMapDataset
from clair_torch.models.icrf_model import ICRFModelDirect
from clair_torch.training.icrf_training import train_icrf
from clair_torch.datasets.collate import custom_collate


def run_full_dataset_training(params: Parameters):

    collate_fn = custom_collate

    # Stage the PairedFrameSettings instances for the image dataset.
    paired_settings, _, _ = file_settings_constructor(dir_paths=params.image_root, file_pattern=f"*{params.image_filetype}",
                                                      recursive=False, default_output_root=params.output_root,
                                                      val_cpu_transforms=params.val_transforms,
                                                      std_cpu_transforms=params.std_transforms, metadata_cls=ImagingMetadata)

    # Stage the image dataset.
    image_dataset = ImageMapDataset(files=paired_settings, missing_std_mode=params.missing_std_mode,
                                    missing_std_value=params.missing_std_value, default_get_item_key="exposure_time")

    # Preload dataset if needed.
    if params.preload_dataset:
        image_dataset.preload_dataset()

    # Stage the model.
    model = ICRFModelDirect(n_points=params.n_points, channels=params.channels,
                            interpolation_mode=params.interpolation_mode,
                            initial_power=params.initial_power).to(params.device)

    # Stage the dataloader.
    dataloader = DataLoader(dataset=image_dataset, batch_size=params.batch_size, shuffle=params.shuffle,
                            drop_last=params.drop_last, pin_memory=params.pin_memory, collate_fn=collate_fn)

    # Stage the optimizer(s).
    if params.one_optimizer_per_channel:
        optimizers = [
            torch.optim.Adam(model.channel_params(c), lr=params.learning_rate, amsgrad=False) for c in range(model.channels)
        ]
    else:
        optimizers = [torch.optim.Adam(torch.stack(model.channel_params(c) for c in range(model.channels)),
                                       lr=params.learning_rate, amsgrad=False)]

    # Stage scheduler for each optimizer.
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50) for optimizer in
        optimizers
    ]

    # Training
    model = train_icrf(dataloader, params.batch_size, params.device, model, optimizers, schedulers, epochs=params.epochs,
                       alpha=params.alpha, beta=params.beta, gamma=params.gamma, delta=params.delta, patience=params.patience,
                       use_relative_linearity_loss=params.use_relative_linearity_loss,
                       use_uncertainty_weighting=params.use_uncertainty_weighting,
                       exposure_ratio_threshold=0.25, lower_valid_threshold=1/255,
                       upper_valid_threshold=254/255)

    save_icrf_txt(model.icrf, Path(f"{params.output_root}/icrf.txt"))

    return


if __name__ == "__main__":
    parameter_dict = cli_parse_args_from_config()
    parameters = Parameters(parameter_dict, strict=True)
    run_full_dataset_training(parameters)
