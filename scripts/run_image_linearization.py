from torch.utils.data import DataLoader
from pathlib import Path
from clair_torch.common import transforms as tr
from clair_torch.common.parameters import Parameters
from clair_torch.common.general_functions import cli_parse_args_from_config
from clair_torch.common.data_io import load_image, load_icrf_txt, save_image
from clair_torch.common.file_settings import file_settings_constructor
from clair_torch.metadata.imaging_metadata import ImagingMetadata
from clair_torch.datasets.image_dataset import ImageMapDataset, FlatFieldArtefactMapDataset, DarkFieldArtefactMapDataset
from clair_torch.datasets.collate import custom_collate
from clair_torch.models.icrf_model import ICRFModelDirect
from clair_torch.inference.linearization import linearize_dataset_generator


def run_image_linearization(params: Parameters):

    """
    if params.darkfield_path is not None:
        bad_pixel_map = load_image(params.darkfield_path, transforms=params.val_transforms)
        bad_pixel_filter_transform = tr.BadPixelCorrection(bad_pixel_map, threshold=5 / 255, kernel_size=5)
    else:
        bad_pixel_filter_transform = None
    """
    bad_pixel_filter_transform = None
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
    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=params.pin_memory,
                            collate_fn=custom_collate)

    # Stage the ICRF model.
    curve = load_icrf_txt(params.icrf_path)
    model = ICRFModelDirect(icrf=curve, interpolation_mode=params.interpolation_mode).to(params.device)

    # Stage the PairedFrameSettings instances for flatfield images.
    paired_flatfield_settings, _, _ = file_settings_constructor(dir_paths=params.flat_field_root,
                                                                file_pattern=f"*.{params.image_filetype}",
                                                                recursive=False, default_output_root=params.output_root,
                                                                val_cpu_transforms=params.val_transforms,
                                                                std_cpu_transforms=params.std_transforms,
                                                                metadata_cls=ImagingMetadata)

    # Stage the flatfield correction dataset.
    flatfield_dataset = FlatFieldArtefactMapDataset(paired_flatfield_settings, True,
                                                    missing_std_mode=params.missing_std_mode,
                                                    missing_std_value=params.missing_std_value,
                                                    attributes_to_match={"magnification": None, "illumination": None})

    paired_dark_field_settings, _, _ = file_settings_constructor(dir_paths=params.dark_field_root,
                                                                 file_pattern=f"*.{params.image_filetype}",
                                                                 recursive=False, default_output_root=params.output_root,
                                                                 val_cpu_transforms=params.val_transforms,
                                                                 std_cpu_transforms=params.std_transforms,
                                                                 metadata_cls=ImagingMetadata)

    dark_field_dataset = DarkFieldArtefactMapDataset(paired_dark_field_settings, True,
                                                     missing_std_mode=params.missing_std_mode,
                                                     missing_std_value=params.missing_std_value,
                                                     attributes_to_match={"exposure_time": 0.3})

    # linearize_dataset_generator is a generator unlike most functions utilizing a dataloader. This makes it easier
    # to handle the single linearized images.
    for i, (linearized, std, meta) in enumerate(linearize_dataset_generator(dataloader, params.device, model,
                                                                            flatfield_dataset=flatfield_dataset,
                                                                            gpu_transforms=None)):

        output_paths = paired_settings[i].get_output_paths()
        if isinstance(output_paths, tuple):
            val_path, std_path = output_paths
        else:
            val_path = output_paths
            std_path = None

        save_image(linearized, val_path)
        if std is not None:
            save_image(std, std_path)


if __name__ == "__main__":
    parameter_dict = cli_parse_args_from_config()
    parameters = Parameters(parameter_dict, strict=True)
    run_image_linearization(parameters)
