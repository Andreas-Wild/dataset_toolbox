"""This module is used to convert label studio masks into numpy arrays saved as PNG images. The module includes a RLEConverter class that can be imported for use elsewhere.
The module may also be run as a script, where the annotations.json, image source directory and, optionally, output directory are provided.
If the output directory is not specified the dataset will be created at the current working directory."""

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .ome_tif.legacy_viewer import LegacyViewer


def _extract_channel(filename: str) -> str:
    """Extract channel subfolder name (e.g. 'channel_1') from a filename like '..._Ch1.ome.tif'."""
    match = re.search(r"Ch(\d+)", filename)
    return f"channel_{match.group(1)}" if match else "unknown_channel"


def _write_channel_output(
    output_path: str, filename: str, img_array, mask_array, metadata: dict
):
    """Write a single channel's image, mask, and metadata to disk (picklable for multiprocessing)."""
    output = Path(output_path)
    channel_dir = _extract_channel(filename)
    cv2.imwrite(
        str(output / "images" / channel_dir / filename.replace(".ome.tif", ".png")),
        img_array,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if mask_array is not None:
        cv2.imwrite(
            str(output / "masks" / channel_dir / filename.replace(".ome.tif", "_mask.png")),
            mask_array,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
    with open(output / "metadata" / channel_dir / filename.replace(".ome.tif", ".json"), "w") as f:
        json.dump(metadata, f, indent=4)


class RLEConverter:
    def __init__(
        self,
        annotation_input_json: Path | str,
        image_source_path: Path | str,
        output_directory: Path | str = Path.cwd(),
    ) -> None:
        self.annotation_json: Path = Path(annotation_input_json)
        self.image_source: Path = Path(image_source_path)
        self.output_directory: Path = Path(output_directory)

    def decode_rle(self, rle, height, width):
        # Convert bytes to a bit array using numpy (much faster than Python-level bit ops)
        bits = np.unpackbits(np.frombuffer(bytes(rle), dtype=np.uint8))

        # Use a simple index tracker into the bit array
        pos = 0

        def read_bits(n):
            nonlocal pos
            # Convert n bits to an integer using positional weighting
            val = int(bits[pos : pos + n].dot(1 << np.arange(n - 1, -1, -1)))
            pos += n
            return val

        num = read_bits(32)
        word_size = read_bits(5) + 1
        rle_sizes = [read_bits(4) + 1 for _ in range(4)]

        out = np.zeros(num, dtype=np.uint8)
        i = 0
        while i < num:
            x = read_bits(1)
            j = i + 1 + read_bits(rle_sizes[read_bits(2)])
            if x:
                val = read_bits(word_size)
                out[i:j] = val
                i = j
            else:
                while i < j:
                    out[i] = read_bits(word_size)
                    i += 1
        return np.reshape(out, [height, width, 4])[:, :, 3]

    def prepare_dataset(self):
        img_dir = os.path.join(self.output_directory, "images")
        mask_dir = os.path.join(self.output_directory, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        with open(self.annotation_json, "r") as f:
            tasks = json.load(f)

        print(f"Total tasks found in JSON: {len(tasks)}")

        processed_count = 0
        skipped_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Converting images...", total=len(tasks)
            )

            for task in tasks:
                # Check if task has annotations
                if "annotations" not in task or not task["annotations"]:
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                # Get the first annotation's result
                annotation = task["annotations"][0]
                if "result" not in annotation or not annotation["result"]:
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                # Extract brush masks from result
                brush_masks = [
                    r for r in annotation["result"] if r.get("type") == "brushlabels"
                ]
                if not brush_masks:
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                # Get image filename from task data
                if "data" not in task or "image" not in task["data"]:
                    print(f"Skipping task {task.get('id')}: No image path found")
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                # Extract filename from URL (handle /data/local-files/?d=filename format)
                image_path = task["data"]["image"]
                if "?d=" in image_path:
                    file_name = image_path.split("?d=")[-1]
                else:
                    file_name = image_path.split("/")[-1]

                # Extract just the base filename (remove any subdirectories)
                base_file_name = os.path.basename(file_name)

                src_path = os.path.join(self.image_source, file_name)

                if not os.path.exists(src_path):
                    print(
                        f"Skipping {file_name}: File not found in {self.image_source}"
                    )
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                try:
                    h = brush_masks[0]["original_height"]
                    w = brush_masks[0]["original_width"]
                except KeyError:
                    print(f"Skipping {file_name}: Missing dimension metadata.")
                    skipped_count += 1
                    progress.advance(task_progress)
                    continue

                # Create Instance Mask (background=0, objects=1,2,3...)
                instance_mask = np.zeros((h, w), dtype=np.uint8)

                for i, mask_data in enumerate(brush_masks):
                    try:
                        rle = mask_data["value"]["rle"]
                        binary_mask = self.decode_rle(rle, h, w)
                        instance_id = i + 1
                        instance_mask[binary_mask > 0] = instance_id
                    except Exception as e:
                        print(f"Error decoding mask for {file_name}: {e}")
                        continue

                # Save Image and Mask as PNG (use base_file_name without subdirs)
                shutil.copy(src_path, os.path.join(img_dir, base_file_name))
                mask_filename = os.path.splitext(base_file_name)[0] + "_mask.png"
                cv2.imwrite(
                    os.path.join(mask_dir, mask_filename),
                    instance_mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )

                processed_count += 1
                progress.advance(task_progress)

        print("\nProcessing complete!")
        print(f"Images with masks created: {processed_count}")
        print(f"Tasks skipped (no masks/missing files): {skipped_count}")
        print(f"Dataset saved to: {self.output_directory}")


class OMEConverter:
    """This class converts .ome.tif files to png images."""

    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.legacy_viewer = LegacyViewer(
            input_dir=self.input_path, output_dir=self.output_path
        )

    def _find_channel_numbers(self) -> set[int]:
        """Scan input files to discover all unique channel numbers."""
        channel_numbers: set[int] = set()
        for filename in self.input_path.glob("*Ch*.ome.tif"):
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
        self._create_dirs()
        image_channel_gen = self.legacy_viewer.get_image_channels()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Count total files first
            ome_files = list(self.input_path.glob("*.ome.tif"))
            task_progress = progress.add_task(
                "[cyan]Converting OME-TIF files...", total=len(ome_files)
            )

            output_str = str(self.output_path)
            max_workers = os.cpu_count() or 4
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for base_filename in image_channel_gen:
                    for filename, img_array, mask_array, metadata in base_filename:
                        futures.append(
                            executor.submit(
                                _write_channel_output,
                                output_str,
                                filename,
                                img_array,
                                mask_array,
                                metadata,
                            )
                        )
                    progress.advance(task_progress)
                # Wait for all writes to finish and raise any errors
                for fut in futures:
                    fut.result()


if __name__ == "__main__":
    # The module can also be run from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "annotation_json",
        type=str,
        help="The annotation.json file returned by Label Studio.",
    )
    parser.add_argument(
        "image_source_directory",
        type=str,
        help="The directory where raw image files are saved. (Typically, label-studio/files/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="The output location where the dataset will be saved.",
    )
    args = parser.parse_args()

    converter = RLEConverter(
        args.annotation_json, args.image_source_directory, args.output_dir
    )
    converter.prepare_dataset()
