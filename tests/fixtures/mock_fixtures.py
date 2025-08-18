import pytest
from unittest.mock import MagicMock, patch

from typing import Optional, Callable, Sequence
from pathlib import Path

from tests.fixtures.base_fixtures import numpy_array, temp_image_file

from clair_torch.common import file_settings as fs
from clair_torch.metadata.base import BaseMetadata


class FakeMetadata(BaseMetadata):
    def __init__(self, values: dict):
        self._values = values

    @property
    def _numeric_fields(self) -> list[str]:
        pass

    @property
    def _text_fields(self) -> list[str]:
        pass

    @property
    def values(self):
        return self._values

    def get_numeric_metadata(self) -> dict[str, float]:
        return {k: v for k, v in self._values.items() if isinstance(v, (int, float))}

    def get_text_metadata(self) -> dict[str, str | None]:
        return {k: v for k, v in self._values.items() if isinstance(v, str)}

    def is_match(self, other: 'FakeMetadata', attributes) -> bool:
        return all(self._values.get(attr) == other._values.get(attr) for attr in attributes)


class FakeFrameSettings:
    def __init__(self, metadata_dict: dict, val_input_path: Optional[Path] = None,
                 std_input_path: Optional[Path] = None, transforms: Optional[Callable] = None):
        self.metadata = FakeMetadata(metadata_dict)
        self.val_input_path = val_input_path
        self.std_input_path = std_input_path
        self.transforms = transforms

    def get_input_paths(self):
        return self.val_input_path, self.std_input_path

    def get_numeric_metadata(self):
        return self.metadata.get_numeric_metadata()

    def get_text_metadata(self):
        return self.metadata.get_text_metadata()

    def get_all_metadata(self):
        return self.metadata.values

    def get_transforms(self):
        return self.transforms

    def is_match(self, other: 'FakeFrameSettings', attributes: Sequence[str]):
        return self.metadata.is_match(other.metadata, attributes)


@pytest.fixture
def fake_metadata_factory():
    def _create(values: dict):
        return FakeMetadata(values)
    return _create


@pytest.fixture
def fake_frame_settings_factory():
    def _create(values: dict, val_input_path: Optional[Path] = None, std_input_path: Optional[Path] = None,
                transforms: Optional[Callable] = None):
        return FakeFrameSettings(values, val_input_path, std_input_path, transforms)
    return _create


@pytest.fixture
def frame_settings_with_mock_metadata(numpy_array, temp_image_file):
    img_path = temp_image_file(numpy_array)
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)

    with patch('clair_torch.common.file_settings.issubclass', return_value=True):
        frame_settings = fs.FrameSettings(input_path=img_path,
                                          output_path=None,
                                          default_output_root=None,
                                          metadata_cls=mock_cls)
    return frame_settings, mock_instance, mock_cls, img_path


@pytest.fixture
def mock_metadata_cls():
    instances = []

    def factory(*args, **kwargs):
        instance = MagicMock()
        instances.append(instance)
        return instance

    mock_cls = MagicMock(side_effect=factory)
    mock_cls.instances = instances  # attach for test access
    return mock_cls


