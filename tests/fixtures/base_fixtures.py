from typing import Optional, Callable
import pytest
from pathlib import Path
import random
import string

import tempfile

import numpy as np
import cv2 as cv
import torch

from clair_torch.common.enums import ChannelOrder
from clair_torch.common import transforms as tr


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], scope="session")
def device_fixture(request):
    return torch.device(request.param)


@pytest.fixture
def channel_order():
    return ChannelOrder.RGB


@pytest.fixture
def numpy_array(request: Optional = None):
    if request is None:
        return np.ones((5, 5)).astype(np.uint8)
    else:
        return request.param


@pytest.fixture
def temp_icrf_txt_file(request, tmp_path) -> Path:
    icrf_data = request.param
    txt_path = tmp_path / "icrf.txt"
    np.savetxt(txt_path, icrf_data)
    return txt_path


@pytest.fixture
def temp_image_file(tmp_path) -> Callable:
    def _create(numpy_array: np.ndarray = np.ones((2, 2)).astype(np.uint8),
                name: str = "image.tif") -> Path:
        img_path = tmp_path / name
        cv.imwrite(str(img_path), numpy_array.astype(np.float64))
        return img_path
    return _create


@pytest.fixture
def generate_temp_file_tree(tmp_path):
    def _create_tree(depth: int, num_files_per_level: int, file_suffix: str) -> Path:
        def make_random_image(path: Path):
            data = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
            cv.imwrite(str(path), data)

        def make_random_text(path: Path):
            content = ''.join(random.choices(string.ascii_letters + string.digits, k=100))
            with open(path, 'w') as f:
                f.write(content)

        def make_file(path: Path):
            if file_suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                make_random_image(path)
            else:
                make_random_text(path)

        current_dir = tmp_path
        for level in range(depth):
            # Create files at this level
            for i in range(num_files_per_level):
                file_path = current_dir / f"file_{level}_{i}{file_suffix}"
                make_file(file_path)
            # Create one subdirectory and go deeper
            current_dir = current_dir / f"subdir_{level}"
            current_dir.mkdir()

        return tmp_path  # Return root of structure

    return _create_tree


@pytest.fixture
def generate_temp_paired_files(tmp_path):
    def _create(file_suffix=".tif", num_paired=2, num_main=1, num_std=1, write_file=False) -> list[Path]:
        paths = []

        def optional_write(p: Path):
            if write_file:
                p.write_text("")  # Create an empty file

        # Paired files (main + STD)
        for i in range(num_paired):
            base_name = f"file{i}"
            main_path = tmp_path / f"{base_name}{file_suffix}"
            std_path = tmp_path / f"{base_name} STD{file_suffix}"
            optional_write(main_path)
            optional_write(std_path)
            paths.extend([main_path, std_path])

        # Additional unpaired main files
        for i in range(num_main):
            main_path = tmp_path / f"main_extra_{i}{file_suffix}"
            optional_write(main_path)
            paths.append(main_path)

        # Additional unpaired STD files
        for i in range(num_std):
            std_path = tmp_path / f"std_extra_{i} STD{file_suffix}"
            optional_write(std_path)
            paths.append(std_path)

        return paths

    return _create
