from matplotlib.pyplot import savefig, clf
from torch.utils.data import DataLoader

from clair_torch.models.icrf_model import ICRFModelDirect
from clair_torch.common.parameters import Parameters
from clair_torch.common.general_functions import cli_parse_args_from_config
from clair_torch.datasets.image_dataset import ImageMapDataset
from clair_torch.datasets.collate import custom_collate
from clair_torch.common import transforms as tr, data_io
from clair_torch.common.file_settings import file_settings_constructor
from clair_torch.metadata.imaging_metadata import ImagingMetadata
from clair_torch.inference.measure_linearity import measure_linearity
from clair_torch.visualization.plotting import plot_linearity_loss


def run_linearity_measurement(params: Parameters):

    # Stage the PairedFrameSettings instances for the image dataset.
    paired_settings, _, _ = file_settings_constructor(dir_paths=params.image_root,
                                                      file_pattern=f"*{params.image_filetype}",
                                                      recursive=False, default_output_root=params.output_root,
                                                      val_cpu_transforms=params.val_transforms,
                                                      std_cpu_transforms=params.std_transforms, metadata_cls=ImagingMetadata)

    # Stage the image dataset.
    image_dataset = ImageMapDataset(files=paired_settings, missing_std_mode=params.missing_std_mode,
                                    missing_std_value=params.missing_std_value)

    # Stage the dataloader.
    dataloader = DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=params.shuffle,
                            drop_last=params.drop_last, pin_memory=params.pin_memory, collate_fn=custom_collate)

    # Stage the model.
    if params.icrf_path is not None:
        curve = data_io.load_icrf_txt(params.icrf_path)
        model = ICRFModelDirect(icrf=curve, interpolation_mode=params.interpolation_mode).to(params.device)
    else:
        model = None

    # Run linearity measurement
    exposure_time_ratios, spatial_linearity_loss, spatial_linearity_loss_std, spatial_linearity_loss_error = (
        measure_linearity(dataloader, params.device, params.use_uncertainty_weighting,
                          params.use_relative_linearity_loss, model)
    )

    # Visualize the data.
    figure = plot_linearity_loss(exposure_time_ratios, spatial_linearity_loss,
                                 spatial_linearity_loss_std, spatial_linearity_loss_error)

    savefig(params.output_root.joinpath("linearity_loss.png"), dpi=300)
    clf()


if __name__ == "__main__":
    parameter_dict = cli_parse_args_from_config()
    parameters = Parameters(parameter_dict, strict=True)
    run_linearity_measurement(parameters)
