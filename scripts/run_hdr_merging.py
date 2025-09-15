from torch.utils.data import DataLoader

from clair_torch.common.parameters import Parameters
from clair_torch.common.general_functions import cli_parse_args_from_config
from clair_torch.common.file_settings import file_settings_constructor
from clair_torch.datasets.image_dataset import ImageMapDataset, FlatFieldArtefactMapDataset, DarkFieldArtefactMapDataset
from clair_torch.models.icrf_model import ICRFModelDirect
from clair_torch.common.data_io import load_icrf_txt, save_image
from clair_torch.training.losses import gaussian_value_weights
from clair_torch.inference.hdr_merge import compute_hdr_image
from clair_torch.datasets.collate import custom_collate
from clair_torch.metadata.imaging_metadata import ImagingMetadata

from clair_torch.common.enums import DarkFieldMode, MissingValMode


def run_hdr_merging(params: Parameters):

    # Stage the transform operations to be performed right after loading an image.
    val_transforms = params.val_transforms
    std_transforms = params.std_transforms
    gpu_transforms = params.gpu_transforms

    # Stage the PairedFrameSettings instances for the image dataset.
    paired_settings, _, _ = file_settings_constructor(dir_paths=params.image_root, file_pattern=f"*{params.image_filetype}",
                                                      recursive=False, default_output_root=params.output_root,
                                                      val_cpu_transforms=val_transforms,
                                                      std_cpu_transforms=std_transforms, metadata_cls=ImagingMetadata)

    # Stage the image dataset.
    image_dataset = ImageMapDataset(files=paired_settings, missing_std_mode=params.missing_std_mode,
                                    missing_std_value=params.missing_std_value)

    # Stage the dataloader.
    dataloader = DataLoader(dataset=image_dataset, batch_size=params.batch_size, shuffle=params.shuffle,
                            drop_last=params.drop_last, pin_memory=params.pin_memory, collate_fn=custom_collate)

    # Stage the model.
    if params.icrf_path is not None:
        curve = load_icrf_txt(params.icrf_path)
        model = ICRFModelDirect(icrf=curve, interpolation_mode=params.interpolation_mode).to(params.device)
    else:
        model = None

    # Stage the PairedFrameSettings instances for flatfield images.
    paired_flatfield_settings, _, _ = file_settings_constructor(dir_paths=params.flat_field_root,
                                                                file_pattern=f"*{params.image_filetype}", recursive=False,
                                                                default_output_root=params.output_root,
                                                                val_cpu_transforms=val_transforms,
                                                                std_cpu_transforms=std_transforms,
                                                                metadata_cls=ImagingMetadata)

    paired_darkfield_settings, _, _ = file_settings_constructor(dir_paths=params.dark_field_root,
                                                                file_pattern=f"*{params.image_filetype}", recursive=False,
                                                                default_output_root=params.output_root,
                                                                val_cpu_transforms=val_transforms,
                                                                std_cpu_transforms=std_transforms,
                                                                metadata_cls=ImagingMetadata)

    # Stage the flatfield correction dataset.
    flatfield_dataset = FlatFieldArtefactMapDataset(paired_flatfield_settings, True,
                                                    missing_std_mode=params.missing_std_mode,
                                                    missing_std_value=params.missing_std_value,
                                                    attributes_to_match={"magnification": None, "illumination": None})

    darkfield_dataset = DarkFieldArtefactMapDataset(paired_darkfield_settings, True,
                                                    missing_std_mode=params.missing_std_mode,
                                                    missing_std_value=params.missing_std_value,
                                                    attributes_to_match={"exposure_time": 0.3},
                                                    dark_field_mode=DarkFieldMode.CLOSEST,
                                                    cache_size=0, missing_val_mode=MissingValMode.SKIP_BATCH)

    # Grab output paths from the first PairedFrameSettings.
    val_output_path, std_output_path = paired_settings[0].get_output_paths()

    # HDR merging.
    hdr_val, hdr_std = compute_hdr_image(dataloader, params.device, model, weight_fn=gaussian_value_weights,
                                         flat_field_dataset=flatfield_dataset,
                                         gpu_transforms=None, dark_field_dataset=darkfield_dataset)

    save_image(hdr_val, val_output_path)
    save_image(hdr_std, std_output_path)


if __name__ == "__main__":
    parameter_dict = cli_parse_args_from_config()
    parameters = Parameters(parameter_dict, strict=True)
    run_hdr_merging(parameters)

