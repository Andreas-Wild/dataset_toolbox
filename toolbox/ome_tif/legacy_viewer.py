"""The legacy_viewer class is used to handle IDEAS' legacy .ome.tiff file format and rewrite it into a modern schema.
    Primary improvements include:
    - Update outdated xml schema.
    - Rewrite multiple single channel images into a single multi channel file."""

import numpy as np
import tifffile
from pathlib import Path
from typing import Generator
from rich import print
import warnings
import logging

# Suppress tifffile warnings for malformed legacy TIFF files
warnings.filterwarnings('ignore', module='tifffile')
logging.getLogger('tifffile').setLevel(logging.ERROR)

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
            yield str(filename).strip("_Ch1.ome.tif")
    
    def _read_pgm_mask(self, file_path: str) -> np.ndarray:
        """Read a PGM (P2 ASCII format) mask file and return as numpy array."""
        with open(file_path, 'r') as f:
            # Read and validate magic number
            magic = f.readline().strip()
            if magic != 'P2':
                raise ValueError(f"Unsupported PGM format: {magic}")
            
            # Skip comment lines
            line = f.readline()
            while line.startswith('#'):
                line = f.readline()
            
            # Parse dimensions (width height)
            width, height = map(int, line.split())
            
            # Parse max value, since it is a boolean array we don't use it.
            max_val = int(f.readline())
            
            # Read all pixel data
            data_text = f.read()
            data = np.array(data_text.split(), dtype=np.uint8)
            
        return data.reshape((height, width))
    
    def get_image_channels(self, dtype:np.uint8|np.uint16=np.uint8) -> Generator[list[tuple[np.ndarray, np.ndarray, dict]]]:
        """Return the different image channels with their calculated default mask. Returns a generator that yields all image channels per base filename."""
        for file_path in self.filename_generator:
            image_channels = []
            for channel in ["_Ch1", "_Ch6"]:
                try:
                    # Read the tiffile and extract metadata:
                    filename = file_path + channel + ".ome.tif"
                    with tifffile.TiffFile(filename) as tif_file:
                        image_array = tif_file.asarray().astype(dtype)
                        metadata_dict = tifffile.xml2dict(tif_file.ome_metadata)
                    # Read the corresponding mask using numpy
                    mask_path = file_path.replace("data/", "data/ExportedMasks/") + channel + ".dmask.pgm"
                    mask_array = self._read_pgm_mask(mask_path).astype(dtype)
                    image_channels.append((filename, image_array, mask_array, metadata_dict))
                except FileNotFoundError:
                    print(f"File {file_path} not found. Skipping...")
                    continue
                except Exception as e:
                    print(f"Error reading tiff file: {e}")
                    continue
            yield image_channels