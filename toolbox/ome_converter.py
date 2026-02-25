"""This class is used to convert .ome.tif images with their accompanying .dmask.pgm file to png images.
The images are also conveniently grouped by channel number."""

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile
from numpy.typing import NDArray

# Specific type definition for UINT8 and UINT16
IntDType = type[np.uint8] | type[np.uint16]

logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="tifffile")


def _normalize_to_dtype(
    image: NDArray[np.uint8 | np.uint16], dtype: IntDType
) -> NDArray:
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


def _read_pgm_mask(file_path: str) -> NDArray:
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
    file_path: Path, channel: int, dtype: IntDType
) -> tuple[str, NDArray, NDArray | None, dict] | None:
    """Read a single channel's image, mask, and metadata as a tuple. Returns None on failure.

    ``file_path`` is a concrete path whose channel suffix will be replaced
    with *channel* so that one "base" path can be reused across channels.
    """
    # Build the actual path for this channel by replacing the Ch<N> portion
    channel_path = Path(re.sub(r"Ch\d+", f"Ch{channel}", str(file_path)))
    try:
        with tifffile.TiffFile(channel_path) as tif_file:
            image_array = _normalize_to_dtype(tif_file.asarray(), dtype)
            if tif_file.ome_metadata is not None:
                metadata_dict = tifffile.xml2dict(tif_file.ome_metadata)
            else:
                metadata_dict = {}

        # Look for mask in an ExportedMasks/ sibling folder
        # Mask filename matches image but with .dmask.pgm instead of .ome.tif
        # e.g. buffy_BR127_20260206_1449_1_Ch1.dmask.pgm
        mask_name = channel_path.name.replace(".ome.tif", ".dmask.pgm")
        mask_path = channel_path.parent / "ExportedMasks" / mask_name

        try:
            mask_array = _normalize_to_dtype(_read_pgm_mask(str(mask_path)), dtype)
        except FileNotFoundError:
            mask_array = None

        return (channel_path.name, image_array, mask_array, metadata_dict)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading tiff file: {e}")
        return None


def _write_channel_output(
    output_path: str, filename: str, img_array, mask_array, metadata: dict
):
    """Write a single channel's image, mask, and metadata to disk (picklable for multiprocessing)."""
    output = Path(output_path)
    match = re.search(r"Ch(\d+)", filename)
    channel_dir = f"channel_{match.group(1)}" if match else "channel_unknown"
    cv2.imwrite(
        str(output / "images" / channel_dir / filename.replace(".ome.tif", ".png")),
        img_array,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if mask_array is not None:
        cv2.imwrite(
            str(
                output
                / "masks"
                / channel_dir
                / filename.replace(".ome.tif", "_mask.png")
            ),
            mask_array,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
    with open(
        output / "metadata" / channel_dir / filename.replace(".ome.tif", ".json"), "w"
    ) as f:
        json.dump(metadata, f, indent=4)


class OMEConverter:
    """This class converts .ome.tif files to png images.

    Can also be used in read-only mode (without an output_path) to browse
    images via :meth:`get_filenames` and :meth:`load_channels`.
    """

    def __init__(self, input_path: str | Path, output_path: str | Path | None = None):
        # First handle input path and check that it is a valid directory
        self.input_path = Path(input_path).expanduser()
        if not self.input_path.is_dir():
            raise NotADirectoryError(
                f"The input path {self.input_path} is not a valid directory. Double check and try again!"
            )
        # Parse output path to be a path (None when used as a viewer)
        self.output_path = Path(output_path).expanduser() if output_path else None
        # Get the channel numbers once on init
        self.channel_numbers: list[int] = self._find_channel_numbers()
        # Build a sorted list of base filenames (one per image, using the first channel)
        self.filenames: list[Path] = sorted(
            self.input_path.glob(f"*Ch{self.channel_numbers[0]}.ome.tif")
        ) if self.channel_numbers else []

    def _find_channel_numbers(self, n: int = 20) -> list[int]:
        """Scan first n input files to discover all unique channel numbers."""
        channel_numbers: set[int] = set()
        for filename in list(self.input_path.glob("*.ome.tif"))[:n]:
            match = re.search(r"Ch(\d+)", filename.name)
            if match:
                channel_numbers.add(int(match.group(1)))
        return list(channel_numbers)

    def _create_dirs(self):
        for subdir in ["images", "masks", "metadata"]:
            for ch in self.channel_numbers:
                self.output_path.joinpath(subdir, f"channel_{ch}").mkdir(
                    parents=True, exist_ok=True
                )

    def get_filenames(self) -> list[str]:
        """Return a list of base image names (one per image set, using the first channel)."""
        return [f.name for f in self.filenames]

    def load_channels(
        self, index: int, dtype: IntDType = np.uint8
    ) -> list[tuple[str, NDArray, NDArray | None, dict]]:
        """Load all channels for the image at *index*.

        Returns a list of (filename, image_array, mask_array, metadata) tuples,
        one per channel.
        """
        base_path = self.filenames[index]
        results: list[tuple[str, NDArray, NDArray | None, dict]] = []
        for channel in self.channel_numbers:
            result = _read_channel(file_path=base_path, channel=channel, dtype=dtype)
            if result:
                results.append(result)
        return results

    def convert(self):
        # First check that we did actually find files with channel numbers
        if not self.channel_numbers:
            print(
                "No channels could be identified. Ensure that fluorescent channels are denoted by Ch<channel_number>."
            )
            print("Skipping conversion...")
            return

        if self.output_path is None:
            print("No output path set. Skipping conversion...")
            return

        # create the necessarry directories to save files to
        try:
            self._create_dirs()
        except PermissionError:
            print(
                "The program does not have permission to create directories at this output location."
            )
            print("Skipping conversion...")
            return
        except Exception as e:
            print("An unknown error occured!")
            print(e)

        for filename in self.filenames:
            for channel in self.channel_numbers:
                result = _read_channel(
                    file_path=filename, channel=channel, dtype=np.uint8
                )
                if result:
                    name, image, mask, metadata = result
                    _write_channel_output(
                        str(self.output_path), name, image, mask, metadata
                    )
                else:
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="Path to the input directory. Should contain .ome.tif files and ExportedMasks directory.",
    )
    parser.add_argument("output_path", help="Path to the output directory")
    args = parser.parse_args()
    # Create converter object
    converter = OMEConverter(input_path=args.input_path, output_path=args.output_path)
    # Convert the images
    converter.convert()
