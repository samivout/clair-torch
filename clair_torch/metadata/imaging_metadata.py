from pathlib import Path
import re

from typeguard import typechecked

from clair_torch.metadata.base import BaseMetadata
from clair_torch.common.data_io import _get_frame_count


class ImagingMetadata(BaseMetadata):
    """
    Class for managing the metadata of an optical microscope image, containing metadata fields for exposure time,
    magnification, illumination type and subject name. The metadata is parsed based on the file name, which is assumed
    to have the following parts, in arbitrary order, for successful parsing. White space is reserved for separating
    parts.
    1. Exposure time in milliseconds with a point for decimal separator, ending in ms with no white space. e.g. '10.5ms'.
    2. Magnification as float with a point for decimal separator, ending in 'x' or 'X' with no white space. e.g. '50.0x'.
    3. Illumination type is reserved as 'BF' or 'bf' for bright field and 'DF' or 'df' for dark field.
    4. Subject name will be parsed as the first part that doesn't fit into any of the aforementioned categories.
    """
    @typechecked
    def __init__(self, val_input_path: str | Path):
        """
        Initialization of a ImagingMetadata instance. Metadata parsing based on the file name.
        Args:
            val_input_path: the path to the file for which to parse metadata.
        """
        if isinstance(val_input_path, str):
            val_input_path = Path(val_input_path)
        if not isinstance(val_input_path, Path):
            raise TypeError(f"Expected val_input_path as Path, got {type(val_input_path)}")

        self.exposure_time = None
        self.magnification = None
        self.illumination = None
        self.subject = None

        self._parse_file_name(val_input_path)

    @property
    def _text_fields(self) -> list[str]:
        return ["illumination", "subject"]

    @property
    def _numeric_fields(self) -> list[str]:
        return ["exposure_time", "magnification"]

    def _parse_file_name(self, val_input_path: Path):
        """
        Extracts metadata fields from the filename. Attempts to parse exposure time, magnification, illumination type
        and subject name. After successful parsing the parsed value is assigned to instance attributes.
        """
        file_name_array = val_input_path.stem.split()

        for element in file_name_array:
            lower_elem = element.casefold()

            # Try exposure time
            if self.exposure_time is None and re.match(r"^\d+.*ms$", element):
                try:
                    self.exposure_time = float(element.removesuffix('ms')) / 1000
                    continue
                except ValueError:
                    pass

            # Try magnification
            if self.magnification is None and re.match(r"^\d+.*[xX]$", element):
                try:
                    self.magnification = float(element.lower().removesuffix('x'))
                    continue
                except ValueError:
                    pass

            # Try illumination
            if self.illumination is None and lower_elem in {'bf', 'df'}:
                self.illumination = element.lower()
                continue

            # If none match and subject is not yet set
            if self.subject is None:
                self.subject = element


class VideoMetadata(ImagingMetadata):
    """
    Class for managing the metadata of a video file, based off of the ImagingMetadata class. Additional feature to that
    is the numeric metadata field 'number_of_frames', which is parsed using a function that calls OpenCV to get the
    number of frames.
    """
    @typechecked
    def __init__(self, val_input_path: str | Path):

        super().__init__(val_input_path)
        self.number_of_frames = _get_frame_count(val_input_path)

    @property
    def _numeric_fields(self) -> list[str]:
        return ["exposure_time", "magnification", "number_of_frames"]
