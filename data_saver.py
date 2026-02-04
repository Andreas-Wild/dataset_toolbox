from datetime import datetime
import json
import numpy as np
from PIL import Image
from pathlib import Path


class DatasetSaver:
    """
    The DatasetSaver class is used to save images, masks and metadata with unique ids. This class is the final part of the data pipeline for creating my master's project dataset.
    """
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.image_dir = self.output_path / "images"
        self.mask_dir = self.output_path / "masks"
        self.metadata_file = self.output_path / "metadata.json"

        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize counter
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"counter": 0, "samples": {}}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def is_processed(self, original_name: str) -> bool:
        """Check if an image has already been processed"""
        for sample_data in self.metadata["samples"].values():
            if sample_data.get("original_name") == original_name:
                return True
        return False

    def generate_filename(self, original_name: str | None = None) -> str:
        """Generate unique filename with timestamp and counter"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["counter"] += 1
        counter = self.metadata["counter"]
        filename = f"{timestamp}_{counter:04d}.png"

        # Store mapping if original name provided
        if original_name:
            self.metadata["samples"][filename] = {
                "original_name": original_name,
                "created_at": timestamp,
            }

        return filename

    def save_sample(
        self, image: np.ndarray, mask: np.ndarray, original_name: str | None = None
    ) -> str:
        """Save image and mask pair with unique naming"""
        filename = self.generate_filename(original_name)

        # Save image and mask
        Image.fromarray(image).save(self.image_dir / filename)
        Image.fromarray(mask).save(self.mask_dir / filename)

        self._save_metadata()
        return filename
