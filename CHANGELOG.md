# Changelog

## [Unreleased]

### Added

- Introduced `Typeguard` for runtime type validation, which can be configured on/off via environment variables.
- `DimensionOrder` enum class for controlling the order of dimensions at IO boundaries.

### Fixed

- Broken batch loop in `inferential_statistics.py`

### Changed

- References to `.validation.type_checks.validate_all` removed in favor of `Typeguard` decorator checks.
- In `losses.py` and `ICRFModelBase` and its subclasses, changed the internal channel ordering of the data from (n_points, channels) dimension order to (channels, n_points).

### Removed

- Removed Protocol based Transforms, preferring BaseTransform for type hints and as base for writing Transform classes.

## [0.2.0]

### Added

- CLI scripts as use reference, runnable with a config.yaml file. For example, within your pipenv at project root: 

`python -m .\scripts\run_icrf_model_training.py --config .\scripts\config.yaml`

- Serialization and deserialization for Transform classes.
- New base class, `MultiFileArtefactMapDataset` as basis for image correction datasets.
- New `DarkFieldArtefactMapDataset` class for handling bad pixel correction in place of the removed Transform class.
- `Parameters` class for easy management of input parameter configuration in scripts, along with serialization and deserialization.
- Functions to read and dump .yaml files.
- Functionality for tolerances in numeric metadata equality checks.
- New Enum classes for choosing handling mode of dark field, flat field and value images.
- Tolerance functionality for matching numeric metadata fields.
- New module for mixin classes with a new `Clearable` class for enabling easy resetting of given instance attributes.
- `pyproject.toml` for build management.
- Dev script for easy creation of a new tag and moving the Unreleased changes in `CHANGELOG.md` under a new tag.

### Fixed

- Broken imports in some modules.
- Handling of None GPU Transforms in loops.

### Changed

- Refactored previous `ArtefactMapDataset` into current `FlatFieldArtefactMapDataset` based off of the new base class.
- Refactored BadPixelCorrection Transform class into an ArtefactDataset class.
- `custom_collate` now returns also a tensor of indices based on the indices of the items in the dataset.
- Dataset classes now support a metadata key to access items in specific orders via `__getitem__`, along with a default key definition on initialization.

### Removed

- Removed `BadPixelCorrection` class due to growing complexity, functionality refactored into Dataset classes.


## [0.1.0]

Made the repository public and started version numbering.

### Added

Everything.

### Changed

### Fixed

### Removed