"""This class is used to convert .ome.tif images with their accompanying .dmask.pgm file to png images.
The images are also conveniently grouped by channel number."""

import os
import re
from pathlib import Path

import numpy as np
import tifffile


def _normalize_to_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Normalize an image array to the target dtype's range.

    When converting from a higher bit-depth (e.g. uint16) to a lower one
    (e.g. uint8), a raw `.astype()` truncates by taking ``value % 256``,
    which produces salt-and-pepper artifacts.  Instead, this helper
    rescales the pixel values linearly into the target range.
    """
    if image.dtype == dtype:
        return image
    target_info = np.iinfo(dtype)
    img_min, img_max = float(image.min()), float(image.max())
    if img_max > img_min:
        scaled = (
            (image.astype(np.float64) - img_min) / (img_max - img_min) * target_info.max
        )
        return scaled.astype(dtype)
    return np.zeros_like(image, dtype=dtype)


def _read_pgm_mask(file_path: str) -> np.ndarray:
    """Read a PGM (P2 ASCII format) mask file and return as numpy array."""
    with open(file_path, "r") as f:
        magic = f.readline().strip()
        if magic != "P2":
            raise ValueError(f"Unsupported PGM format: {magic}")
        line = f.readline()
        while line.startswith("#"):
            line = f.readline()
        width, height = map(int, line.split())
        # max_val is read but unused. we simply want to move to next line.
        _ = int(f.readline())
        data = np.fromstring(f.read(), dtype=np.uint8, sep=" ")
    return data.reshape((height, width))


def _read_channel(
    file_path: Path, channel: int, dtype: np.dtype
) -> tuple[str, np.ndarray, np.ndarray | None, dict] | None:
    """Read a single channel's image, mask, and metadata as a tuple. Returns None on failure."""
    try:
        filename = file_path + str(channel) + ".ome.tif"
        with tifffile.TiffFile(filename) as tif_file:
            image_array = _normalize_to_dtype(tif_file.asarray(), dtype)
            if tif_file.ome_metadata is not None:
                metadata_dict = tifffile.xml2dict(tif_file.ome_metadata)
            else:
                metadata_dict = {}

        # Look for mask in an ExportedMasks/ sibling folder
        base_dir = str(Path(file_path).parent)
        base_name = Path(file_path).name
        mask_path = os.path.join(
            base_dir, "ExportedMasks", base_name + str(channel) + ".dmask.pgm"
        )

        try:
            # read the mask file and convert to uint8
            mask_array = _normalize_to_dtype(_read_pgm_mask(mask_path), dtype)
        except FileNotFoundError:
            mask_array = None

        return (filename.split("/")[-1], image_array, mask_array, metadata_dict)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading tiff file: {e}")
        return None


class OMEConverter:
    """This class converts .ome.tif files to png images."""

    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def _find_channel_numbers(self, n: int = 20) -> set[int]:
        """Scan first n input files to discover all unique channel numbers."""
        channel_numbers: set[int] = set()
        for filename in list(self.input_path.glob("*Ch*.ome.tif"))[:n]:
            match = re.search(r"Ch(\d+)", filename.name)
            if match:
                channel_numbers.add(int(match.group(1)))
        return channel_numbers

    def _create_dirs(self):
        channel_numbers = self._find_channel_numbers()
        for subdir in ["images", "masks", "metadata"]:
            for ch in channel_numbers:
                self.output_path.joinpath(subdir, f"channel_{ch}").mkdir(
                    parents=True, exist_ok=True
                )

    def convert(self):
        # create the necessarry directories to save files to
        self._create_dirs()
        # get available channels
        channels = list(self._find_channel_numbers())
        # get a list of all the image filenames use the first channel we know exists
        filenames: list[Path] = list(self.input_path.glob(f"*Ch{channels[0]}.ome.tif"))
        for filename in filenames:
            for channel in channels:
                _read_channel(file_path=filename, channel=channel, dtype=np.uint8)
