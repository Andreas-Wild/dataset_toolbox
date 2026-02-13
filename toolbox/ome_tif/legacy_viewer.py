"""The legacy_viewer class is used to handle IDEAS' legacy .ome.tiff file format and rewrite it into a modern schema.
    Primary improvements include:
    - Update outdated xml schema.
    - Rewrite multiple single channel images into a single multi channel file."""

import os
import numpy as np
import tifffile
from pathlib import Path
from typing import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich import print
import warnings
import logging

# Suppress tifffile warnings for malformed legacy TIFF files
warnings.filterwarnings('ignore', module='tifffile')
logging.getLogger('tifffile').setLevel(logging.ERROR)


# ── Module-level helpers (must be picklable for ProcessPoolExecutor) ──────────

def _read_pgm_mask(file_path: str) -> np.ndarray:
    """Read a PGM (P2 ASCII format) mask file and return as numpy array."""
    with open(file_path, 'r') as f:
        magic = f.readline().strip()
        if magic != 'P2':
            raise ValueError(f"Unsupported PGM format: {magic}")
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        width, height = map(int, line.split())
        max_val = int(f.readline())
        data = np.fromstring(f.read(), dtype=np.uint8, sep=' ')
    return data.reshape((height, width))


def _read_channel(file_path: str, channel: str, dtype) -> tuple[str, np.ndarray, np.ndarray, dict] | None:
    """Read a single channel's image, mask, and metadata. Returns None on failure."""
    try:
        filename = file_path + channel + ".ome.tif"
        with tifffile.TiffFile(filename) as tif_file:
            image_array = tif_file.asarray().astype(dtype)
            metadata_dict = tifffile.xml2dict(tif_file.ome_metadata)
        mask_path = file_path.replace("data/", "data/ExportedMasks/") + channel + ".dmask.pgm"
        mask_array = _read_pgm_mask(mask_path).astype(dtype)
        return (filename.split('/')[-1], image_array, mask_array, metadata_dict)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading tiff file: {e}")
        return None


class LegacyViewer():
    """The LegacyViewer class transforms old OMETIF file format images into new file formats.
    Input:
    - input_dir: str | Path
    - should_save: bool = False
    - output_dir: str | Path | None = None
    - combine_ch: bool = True
    Output:
    - None """
    def __init__(self, input_dir: str|Path, should_save:bool = False, output_dir: str|Path|None = None, combine_ch: bool = True):
        # Set all state variables
        self.input_dir: Path = Path(input_dir)
        self.should_save: bool = should_save
        self.output_dir: Path|None = Path(output_dir) if output_dir is not None else None
        self.combine_ch: bool = combine_ch
        self.filename_generator: Generator[str] = self._iterdir()

    def _iterdir(self, pattern: str = "*_Ch1.ome.tif") -> Generator[Path]:
        for filename in Path.glob(self.input_dir, pattern=pattern):
            yield str(filename).removesuffix("_Ch1.ome.tif")

    def get_image_channels(self, dtype:np.uint8|np.uint16=np.uint8, max_workers:int|None=None) -> Generator[list[tuple[str, np.ndarray, np.ndarray, dict]]]:
        """Return the different image channels with their calculated default mask.
        Returns a generator that yields all image channels per base filename.
        
        Args:
            dtype: Output array dtype (np.uint8 or np.uint16).
            max_workers: Number of processes for parallel work. Defaults to CPU count.
        """
        channels = ["_Ch1", "_Ch6"]
        if max_workers is None:
            max_workers = os.cpu_count() or 4

        # Collect all file paths upfront so we can submit work in bulk
        file_paths = list(self.filename_generator)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all (file, channel) pairs at once
            future_to_key = {}
            for file_path in file_paths:
                for ch in channels:
                    future = executor.submit(_read_channel, file_path, ch, dtype)
                    future_to_key[future] = file_path

            # Collect results grouped by base filename
            results_by_file: dict[str, list] = {fp: [] for fp in file_paths}
            for future in as_completed(future_to_key):
                file_path = future_to_key[future]
                result = future.result()
                if result is not None:
                    results_by_file[file_path].append(result)

        # Yield in original file order
        for file_path in file_paths:
            if results_by_file[file_path]:
                yield results_by_file[file_path]