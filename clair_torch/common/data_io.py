"""Module for input-output operations.

This module contains all the input and output related functionality of the package. The functions in here are called by
other functions, classes etc. in this project.
"""

from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Sequence
import yaml

import torch
import numpy as np
import cv2 as cv

from clair_torch.common.transforms import Transform
from clair_torch.common.general_functions import cv_to_torch, normalize_container
from clair_torch.common.enums import ChannelOrder
from clair_torch.validation.io_checks import validate_input_file_path, is_potentially_valid_file_path
from clair_torch.validation.type_checks import validate_all


def load_icrf_txt(path: str | Path, source_channel_order: ChannelOrder = ChannelOrder.BGR) -> torch.Tensor:
    """
    Utility function for loading an inverse camera response function from a .txt file. Expects a 2D NumPy array
    with shape (N, C), with N representing the number of datapoints
    Args:
        path: path to the text file containing the ICRF data.
        source_channel_order: the order in which the channels are expected to be in the file.

    Returns:

    """
    if isinstance(path, str):
        path = Path(path)

    validate_input_file_path(path, suffix=".txt")

    try:
        data = np.loadtxt(path)  # shape: (256, 3), BGR order
    except Exception as e:
        raise IOError(f'Failed to load NumPy array from {path}: {e}')

    # No change for any and RGB ordering.
    if source_channel_order == ChannelOrder.ANY or source_channel_order == ChannelOrder.RGB:
        data_rgb = data
    # Reverse BGR order into RGB order.
    elif source_channel_order == ChannelOrder.BGR:
        data_rgb = data[:, ::-1].copy()  # shape: (256, 3)
    # Raise value error for unknown channel ordering.
    else:
        raise ValueError(f"Unknown channel order {source_channel_order}:")

    # Convert to PyTorch tensor
    icrf_tensor = torch.from_numpy(data_rgb).float()

    return icrf_tensor


def save_icrf_txt(icrf: torch.Tensor, path: str | Path) -> None:
    """
    Utility function to save an ICRF into a .txt file of the given filepath.
    Args:
        icrf: the ICRF tensor.
        path: the filepath where to save the file.

    Returns:
        None
    """
    if not isinstance(icrf, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(icrf).__name__}")
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected str or Path, got {type(path).__name__}")
    if not is_potentially_valid_file_path(path):
        raise IOError(f"Invalid path for your OS: {path}")

    data = icrf.detach().cpu().numpy()
    data = data[:, ::-1]

    try:
        np.savetxt(path, data)
    except Exception:
        raise IOError(f"Couldn't save data to path {path}")

    return


def load_principal_components(file_paths: List[str | Path]) -> torch.Tensor:
    """
    Loads principal component data from text files, one per color channel. The files in the input paths should be
    ordered in the desired channel order. E.g. for RGB images it should point to the red, green and blue files in order.

    Args:
        file_paths: List of paths to the .txt files (one per channel).

    Returns:
        A torch.Tensor of shape (n_points, n_components, channels).
    """
    pcs = []
    for path in file_paths:
        data = np.loadtxt(path)
        pcs.append(torch.tensor(data, dtype=torch.float32))  # Shape: (n_points, n_components)

    pcs_tensor = torch.stack(pcs, dim=2)  # Shape: (n_points, n_components, channels)
    return pcs_tensor


def load_image(file_path: Path, transforms: Optional[Transform | Iterable[Transform]] = None) -> torch.Tensor:
    """
    Generic function to load a single image from the given path. Allows also the definition of transformations to be
    performed on the image before returning it upstream.
    Args:
        file_path: path to the image file.
        transforms: single Transform or Iterable of Transforms to be performed on the image before returning it.

    Returns:
        The loaded and possibly operated image in Tensor format.
    """
    validate_input_file_path(file_path, suffix=None)

    transforms = normalize_container(transforms)
    validate_all(transforms, Transform, raise_error=True, name="transform")

    try:
        image = cv.imread(str(file_path), cv.IMREAD_UNCHANGED)
        image = torch.from_numpy(image)
        image = cv_to_torch(image)
    except Exception as e:
        raise IOError(f"Failed to load NumPy array from {file_path}: {e}")

    if transforms:
        for transform in transforms:
            image = transform(image)

    return image


def load_video_frames_generator(file_path: str | Path, transforms: Optional[Transform | Iterable[Transform]] = None):
    """
    Function for loading frames from a video file through a generator.
    Args:
        file_path: path to the video file.
        transforms: optional list of transform operations to perform on each frame before yielding them.

    Returns:

    """
    validate_input_file_path(file_path, suffix=None)

    transforms = normalize_container(transforms)
    validate_all(transforms, Transform, raise_error=True, name="transform")

    cap = cv.VideoCapture(str(file_path))
    success, frame = cap.read()

    while success:
        tensor_frame = torch.from_numpy(frame)
        tensor_frame = cv_to_torch(tensor_frame)

        if transforms:
            for transform in transforms:
                tensor_frame = transform(tensor_frame)

        yield tensor_frame
        success, frame = cap.read()

    cap.release()


def _get_frame_count(file_path: Path):
    """
    Utility function to get the number of frames in a video file.
    Args:
        file_path: filepath to the video.

    Returns:
        The number of frames in the video.
    """
    cap = cv.VideoCapture(str(file_path))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


def save_image(tensor: torch.Tensor, image_save_path: str | Path, dtype: np.dtype = np.float64,
               params: Optional[Sequence[int]] = None) -> None:
    """
    Save a PyTorch tensor as a 32-bit float per channel TIFF image.

    Args:
        tensor: A PyTorch tensor of shape (C, H, W) or (H, W), dtype float32.
        image_save_path: Path to save the image.
        dtype: the NumPy datatype to use to save the image.
        params: Sequence of params to pass to OpenCV imwrite function.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}")

    if not isinstance(image_save_path, (str, Path)):
        raise TypeError(f"Expected path as str or Path, got {type(image_save_path).__name__}")

    if not is_potentially_valid_file_path(image_save_path):
        raise IOError(f"Invalid path for your OS: {image_save_path}")

    if params is not None and not all(isinstance(p, int) for p in params):
        raise TypeError("All OpenCV imwrite params must be integers")

    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_save_path.parent.exists():
        raise IOError(f"Couldn't create the directory structure for path {image_save_path}")

    if tensor.is_cuda:
        tensor = tensor.cpu()

    array = tensor.detach().numpy().astype(dtype=dtype)

    # Handle 3-channel (C, H, W) case: convert to (H, W, C) and reorder to BGR
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))  # (H, W, C)
        if array.shape[2] == 3:
            array = array[:, :, [2, 1, 0]]  # RGB → BGR

    success = cv.imwrite(str(image_save_path), array, params or [])
    if not success:
        raise IOError(f"Failed to save image to {image_save_path}")


def _get_file_input_paths_by_pattern(search_root: str | Path, pattern: str = f"*", recursive_search: bool = False) -> List[Path]:
    """
    Utility function to collect all filepaths of the given file suffix. Optionally allows recursive search of the
    subdirectories of the given search_root.
    Args:
        search_root: the root dirpath to search.
        pattern: the suffix of the filetype to search for.
        recursive_search: whether to recursively search subdirectories of the search_root.

    Returns:
        List of found filepaths, empty if no files are found.
    """
    if not isinstance(search_root, (str, Path)):
        raise TypeError(f"Expected path as str or Path, got {type(search_root).__name__}")

    if not isinstance(pattern, str):
        raise TypeError(f"Expected pattern as str, got {type(pattern).__name__}")

    if isinstance(search_root, str):
        search_root = Path(search_root)

    input_paths = []

    if recursive_search:
        target_files = search_root.rglob(pattern)
    else:
        target_files = search_root.glob(pattern)

    for path in target_files:
        if path.is_file():
            input_paths.append(path)

    return input_paths


def _pair_main_and_std_files(
    paths: List[Path], strict_exclusive: bool = True
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, None]], List[Tuple[None, Path]]]:
    """
    Pairs files with associated STD files and sorts them into three categories. In strict mode categories are mutually
    exclusive, while when strict mode is not used, a file can appear in both the paired and either main or STD
    categories.

    Args:
        paths: A list of file paths.
        strict_exclusive: If True, files that are part of a pair will not be included in the unpaired lists.

    Returns:
        A tuple of three lists:
            1. Pairs with both main and STD files.
            2. Files with only main file.
            3. Files with only STD file.
    """
    std_suffix = " STD"

    main_files = []
    std_files = []

    for path in paths:
        if path.stem.endswith(std_suffix):
            std_files.append(path)
        else:
            main_files.append(path)

    std_map = {p.stem[:-len(std_suffix)]: p for p in std_files}
    paired = []
    paired_main = set()
    paired_std = set()

    for main in main_files:
        match = std_map.get(main.stem)
        if match:
            paired.append((main, match))
            paired_main.add(main)
            paired_std.add(match)

    main_only = [(m, None) for m in main_files if not strict_exclusive or m not in paired_main]
    std_only = [(None, s) for s in std_files if not strict_exclusive or s not in paired_std]

    return paired, main_only, std_only


def read_yaml(file_path: Path) -> dict:
    """
    Utility function to read the contents of an .yaml file into a dictionary.
    Args:
        file_path: path to the .yaml file.

    Returns:
        Dictionary of the .yaml contents.
    """
    validate_input_file_path(file_path, suffix="yaml")
    with open(file_path) as f:
        file_content = yaml.safe_load(f)

    return file_content


def dump_yaml_to_file(dictionary: dict, target_path: Path) -> None:
    """
    Utility function to dump dictionary content to an .yaml file.
    Args:
        dictionary: the dictionary to dump.
        target_path: the filepath to which to save the data.
    """
    is_potentially_valid_file_path(target_path)
    with open(target_path, "w") as f:
        yaml.safe_dump(dictionary, f)

    return
