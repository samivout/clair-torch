from torch.utils.data import DataLoader

from clair_torch.common.parameters import Parameters
from clair_torch.common.general_functions import cli_parse_args_from_config
from clair_torch.common.data_io import load_icrf_txt, save_image
from clair_torch.models.icrf_model import ICRFModelDirect
from clair_torch.common.file_settings import file_settings_constructor
from clair_torch.metadata.imaging_metadata import VideoMetadata
from clair_torch.datasets.video_frame_dataset import VideoIterableDataset
from clair_torch.datasets.collate import custom_collate
from clair_torch.inference.inferential_statistics import compute_video_mean_and_std


def compute_mean_and_var_from_single_video(params: Parameters):

    if params.icrf_path is not None:
        icrf_curve = load_icrf_txt(params.icrf_path)
        icrf_model = ICRFModelDirect(icrf=icrf_curve, interpolation_mode=params.interpolation_mode).to(params.device)
    else:
        icrf_model = None

    # Stage the PairedFrameSettings instances for the image dataset.
    _, main_settings, _ = file_settings_constructor(dir_paths=params.image_root, file_pattern=f"*{params.image_filetype}",
                                                    recursive=False, default_output_root=params.output_root,
                                                    val_cpu_transforms=params.val_transforms,
                                                    std_cpu_transforms=params.std_transforms, metadata_cls=VideoMetadata)

    # Stage the image dataset.
    video_dataset = VideoIterableDataset(frame_settings=main_settings, missing_std_mode=params.missing_std_mode,
                                         missing_std_value=params.missing_std_value)

    # Stage the dataloader.
    dataloader = DataLoader(dataset=video_dataset, batch_size=params.batch_size, shuffle=False,
                            drop_last=params.drop_last, pin_memory=params.pin_memory, collate_fn=custom_collate)

    # Grab output paths from the first PairedFrameSettings.
    val_output_path = main_settings[0].get_output_paths().with_suffix(".tif")
    std_output_path = main_settings[0].get_candidate_std_output_path().with_suffix(".tif")

    # Mean and standard uncertainty computation.
    mean, std = compute_video_mean_and_std(dataloader, device=params.device, icrf_model=icrf_model)

    save_image(mean, val_output_path)
    save_image(std, std_output_path)


if __name__ == "__main__":
    parameter_dict = cli_parse_args_from_config()
    parameters = Parameters(parameter_dict, strict=True)
    compute_mean_and_var_from_single_video(parameters)
