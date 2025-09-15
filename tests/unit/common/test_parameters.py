import pytest

import yaml
from pathlib import Path

from clair_torch.common.parameters import Parameters


class TestParameters:
    def test_parameters_roundtrip(self, tmp_path: Path):

        config = {
            "paths": {
                "image_filetype": ".tif",
                "image_root": str(tmp_path / "images"),
                "output_root": str(tmp_path / "results"),
                "flat_field_root": None,
                "dark_field_root": None,
                "icrf_path": None,
            },
            "dataset": {
                "missing_std_mode": "multiplier",
                "missing_std_value": 0.05,
                "preload_dataset": True,
            },
            "dataloader": {
                "batch_size": 4,
                "shuffle": True,
                "drop_last": False,
                "pin_memory": False,
            },
            "model": {
                "n_points": 256,
                "channels": 3,
                "interpolation_mode": "linear",
                "initial_power": 2.5,
            },
            "training": {
                "learning_rate": 0.001,
                "epochs": 1200,
                "alpha": 10.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "patience": 200,
                "use_relative_linearity_loss": True,
                "use_uncertainty_weighting": False,
                "device": "cpu",
                "one_optimizer_per_channel": True,
                "dtype": "float64",
            },
            'transforms': {'gpu_transforms': [],
                           'std_transforms': [],
                           'val_transforms': []},
        }

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with open(yaml_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        params = Parameters(loaded_config, strict=True)

        roundtrip_path = tmp_path / "config_roundtrip.yaml"
        params.to_yaml(roundtrip_path)

        with open(roundtrip_path, "r") as f:
            roundtrip_config = yaml.safe_load(f)

        assert config == roundtrip_config

    @pytest.mark.parametrize("missing_key", ["image_root", "output_root"])
    def test_missing_required_path_raises(self, tmp_path: Path, missing_key: str):

        # Config with valid defaults
        config = {
            "paths": {
                "image_filetype": ".tif",
                "image_root": str(tmp_path / "images"),
                "output_root": str(tmp_path / "results"),
                "flat_field_root": None,
                "dark_field_root": None,
                "icrf_path": None,
            },
            "dataset": {
                "missing_std_mode": "none",
                "missing_std_value": 0.05,
                "preload_dataset": True,
            },
            "dataloader": {
                "batch_size": 4,
                "shuffle": False,
                "drop_last": False,
                "pin_memory": False,
            },
            "model": {
                "n_points": 128,
                "channels": 3,
                "interpolation_mode": "linear",
                "initial_power": 2.0,
            },
            "training": {
                "learning_rate": 0.01,
                "epochs": 10,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "patience": 5,
                "use_relative_linearity_loss": False,
                "use_uncertainty_weighting": False,
                "device": "cpu",
                "one_optimizer_per_channel": False,
                "dtype": "float32",
            },
        }

        # Remove the required path under test
        config["paths"][missing_key] = None

        with pytest.raises(ValueError, match=missing_key):
            Parameters(config, strict=True)

    @pytest.mark.parametrize("optional_key", ["flat_field_root", "dark_field_root", "icrf_path"])
    def test_optional_paths_can_be_none(self, tmp_path: Path, optional_key: str):

        config = {
            "paths": {
                "image_filetype": ".tif",
                "image_root": str(tmp_path / "images"),
                "output_root": str(tmp_path / "results"),
                "flat_field_root": str(tmp_path / "flatfields"),
                "dark_field_root": str(tmp_path / "darkfield.tif"),
                "icrf_path": str(tmp_path / "icrf.txt"),
            },
            "dataset": {
                "missing_std_mode": "none",
                "missing_std_value": 0.05,
                "preload_dataset": True,
            },
            "dataloader": {
                "batch_size": 4,
                "shuffle": False,
                "drop_last": False,
                "pin_memory": False,
            },
            "model": {
                "n_points": 128,
                "channels": 3,
                "interpolation_mode": "linear",
                "initial_power": 2.0,
            },
            "training": {
                "learning_rate": 0.01,
                "epochs": 10,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "patience": 5,
                "use_relative_linearity_loss": False,
                "use_uncertainty_weighting": False,
                "device": "cpu",
                "one_optimizer_per_channel": False,
                "dtype": "float32",
            },
        }

        # Nullify the optional key under test
        config["paths"][optional_key] = None

        params = Parameters(config, strict=True)
        assert getattr(params, optional_key) is None
