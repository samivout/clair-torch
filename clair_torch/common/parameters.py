from pathlib import Path
from enum import Enum
from torch import dtype
from typing import Optional, Any

from clair_torch.common.data_io import dump_yaml_to_file
from clair_torch.common.transforms import BaseTransform, deserialize_transforms, serialize_transforms
from clair_torch.common.enums import MissingStdMode, InterpMode, DTYPE_MAP, REVERSE_DTYPE_MAP


class Parameters:
    def __init__(self, config: dict[str, Any], strict: bool = True):

        self.strict = strict

        # --- REQUIRED PATHS ---
        paths = config.get("paths", {})

        self.image_filetype: str = self._validate_type(paths.get("image_filetype", ".tif"), str, "paths.image_filetype")
        self.image_root: Path = self._validate_path(paths.get("image_root"), required=True, key="paths.image_root")
        self.output_root: Path = self._validate_path(paths.get("output_root"), required=True, key="paths.output_root")

        # --- OPTIONAL PATHS ---
        self.flat_field_root: Optional[Path] = self._validate_path(paths.get("flat_field_root"), required=False, key="paths.flat_field_root")
        self.dark_field_root: Optional[Path] = self._validate_path(paths.get("dark_field_root"), required=False, key="paths.dark_field_root")
        self.icrf_path: Optional[Path] = self._validate_path(paths.get("icrf_path"), required=False, key="paths.icrf_path")

        # --- DATASET ---
        dataset = config.get("dataset", {})
        self.missing_std_mode: MissingStdMode = self._parse_enum(dataset.get("missing_std_mode", "multiplier"), MissingStdMode, "dataset.missing_std_mode")
        self.missing_std_value: float = self._validate_type(dataset.get("missing_std_value", 0.05), (float, int), "dataset.missing_std_value")
        self.preload_dataset: bool = self._validate_type(dataset.get("preload_dataset", True), bool, "dataset.preload_dataset")

        # --- DATALOADER ---
        dataloader = config.get("dataloader", {})
        self.batch_size: int = self._validate_type(dataloader.get("batch_size", 4), int, "dataloader.batch_size")
        self.shuffle: bool = self._validate_type(dataloader.get("shuffle", True), bool, "dataloader.shuffle")
        self.drop_last: bool = self._validate_type(dataloader.get("drop_last", False), bool, "dataloader.drop_last")
        self.pin_memory: bool = self._validate_type(dataloader.get("pin_memory", False), bool, "dataloader.pin_memory")

        # --- MODEL ---
        model = config.get("model", {})
        self.n_points: int = self._validate_type(model.get("n_points", 256), int, "model.n_points")
        self.channels: int = self._validate_type(model.get("channels", 3), int, "model.channels")
        self.interpolation_mode: InterpMode = self._parse_enum(model.get("interpolation_mode", "linear"), InterpMode, "model.interpolation_mode")
        self.initial_power: float = self._validate_type(model.get("initial_power", 2.5), (float, int), "model.initial_power")

        # --- TRAINING ---
        training = config.get("training", {})
        self.learning_rate: float = self._validate_type(training.get("learning_rate", 0.001), (float, int), "training.learning_rate")
        self.epochs: int = self._validate_type(training.get("epochs", 1200), int, "training.epochs")
        self.alpha: float = self._validate_type(training.get("alpha", 10.0), (float, int), "training.alpha")
        self.beta: float = self._validate_type(training.get("beta", 1.0), (float, int), "training.beta")
        self.gamma: float = self._validate_type(training.get("gamma", 1.0), (float, int), "training.gamma")
        self.delta: float = self._validate_type(training.get("delta", 1.0), (float, int), "training.delta")
        self.patience: int = self._validate_type(training.get("patience", 200), int, "training.patience")
        self.use_relative_linearity_loss: bool = self._validate_type(training.get("use_relative_linearity_loss", True), bool, "training.use_relative_linearity_loss")
        self.use_uncertainty_weighting: bool = self._validate_type(training.get("use_uncertainty_weighting", False), bool, "training.use_uncertainty_weighting")
        self.device: str = self._validate_type(training.get("device", "cpu"), str, "training.device")
        self.one_optimizer_per_channel: bool = self._validate_type(training.get("one_optimizer_per_channel", True), bool, "training.one_optimizer_per_channel")
        self.dtype: dtype = DTYPE_MAP[self._validate_type(training.get("dtype", "float64"), str, "training.dtype")]

        # --- TRANSFORMS ---
        transforms_cfg = config.get("transforms", {})
        self.val_transforms: list[BaseTransform] = deserialize_transforms(transforms_cfg.get("val_transforms", []))
        self.std_transforms: list[BaseTransform] = deserialize_transforms(transforms_cfg.get("std_transforms", []))
        self.gpu_transforms: list[BaseTransform] = deserialize_transforms(transforms_cfg.get("gpu_transforms", []))

    def _validate_type(self, value, expected_type, key: str):
        if not isinstance(value, expected_type):
            if self.strict:
                raise TypeError(f"Config key '{key}' expected type {expected_type}, got {type(value)}.")
            return expected_type() if not isinstance(expected_type, tuple) else expected_type[0]()  # default empty
        return value

    @staticmethod
    def _validate_path(value: Optional[str], required: bool, key: str) -> Optional[Path]:
        if value is None:
            if required:
                raise ValueError(f"Config key '{key}' is required but missing or null.")
            return None
        return Path(value)

    def _parse_enum(self, value: str, enum_cls: Enum, key: str):
        if not isinstance(value, str):
            if self.strict:
                raise TypeError(f"Config key '{key}' must be a string for enum {enum_cls.__name__}.")
            return list(enum_cls)[0]

        value = value.strip().lower()
        mapping = {e.name.lower(): e for e in enum_cls}
        if value not in mapping:
            if self.strict:
                raise ValueError(f"Invalid value '{value}' for {key}. Allowed: {list(mapping.keys())}")
            return list(enum_cls)[0]
        return mapping[value]

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters into a nested dictionary."""
        def path_or_none(p): return str(p) if p is not None else None

        return {
            "paths": {
                "image_filetype": self.image_filetype,
                "image_root": str(self.image_root),
                "flat_field_root": path_or_none(self.flat_field_root),
                "dark_field_root": path_or_none(self.dark_field_root),
                "output_root": str(self.output_root),
                "icrf_path": path_or_none(self.icrf_path),
            },
            "dataset": {
                "missing_std_mode": self.missing_std_mode.name.lower(),
                "missing_std_value": self.missing_std_value,
                "preload_dataset": self.preload_dataset,
            },
            "dataloader": {
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "drop_last": self.drop_last,
                "pin_memory": self.pin_memory,
            },
            "model": {
                "n_points": self.n_points,
                "channels": self.channels,
                "interpolation_mode": self.interpolation_mode.name.lower(),
                "initial_power": self.initial_power,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
                "patience": self.patience,
                "use_relative_linearity_loss": self.use_relative_linearity_loss,
                "use_uncertainty_weighting": self.use_uncertainty_weighting,
                "device": self.device,
                "one_optimizer_per_channel": self.one_optimizer_per_channel,
                "dtype": REVERSE_DTYPE_MAP[self.dtype],
            },
            "transforms": {
                "val_transforms": serialize_transforms(self.val_transforms),
                "std_transforms": serialize_transforms(self.std_transforms),
                "gpu_transforms": serialize_transforms(self.gpu_transforms)
            },
        }

    def to_yaml(self, filepath: Path):
        """Export parameters to a YAML string or file."""
        data = self.to_dict()
        dump_yaml_to_file(data, filepath)
