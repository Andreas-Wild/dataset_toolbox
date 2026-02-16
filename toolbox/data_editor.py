"""This module provides a NiceGUI web app to interactively edit masks. The DatasetEditor class may be imported, and pointed to an arbitrary dataset location,
provided that the location contains a 'images' and 'masks' directory and a 'metadata.json' file. Note that the class may only be initialised inside a nicegui.ui context.
The class allows users to add/remove pixels, grow/shrink masks, merge/separate nearby masks, undo edits and save the result to a new dataset."""

import base64
import io
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from nicegui import ui
from nicegui.events import MouseEventArguments
from .mask_utils import pixel_colour, remove_pixel, merge_mask, grow_mask, shrink_mask, split_connected
from .data_saver import DatasetSaver

# TODO: add toggle to only load files not already handled to avoid repeated clicking through correct images.
# This should be done once per instance to still allow users to replace previous incorrect labels.
class DatasetEditor:
    def __init__(self, dataset_path: Path = Path.cwd() / "input", output_path: Path | str = Path.cwd() / "output"):
        self.dataset_path: Path = Path(dataset_path)
        self.output_path: Path = Path(output_path)
        self.samples: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.current_index: int = 0
        self.current_mask: Optional[np.ndarray] = None
        self.last_click: Optional[tuple[int, int]] = None
        self.mask_history: list[np.ndarray] = []
        self.interactive_image: ui.interactive_image = ui.interactive_image()
        self.status_label: ui.label = ui.label("(X, Y) = (0, 0) | Mask ID: background")
        self.nav_label: ui.label = ui.label()
        self.saver: Optional[DatasetSaver]  = None

    def load_dataset(self) -> int:
        """Load all images and masks from dataset."""
        image_dir = self.dataset_path / "images"
        mask_dir = self.dataset_path / "masks"

        self.samples = []
        for file in sorted(image_dir.glob("*.png")):
            filename = file.name
            try:
                img = np.asarray(Image.open(image_dir / filename))
                mask = np.asarray(Image.open(mask_dir / filename))
                self.samples.append((filename, img, mask))
            except FileNotFoundError:
                ui.notify(f"Missing mask for: {filename}", type="warning")

        if self.samples:
            self.current_index = 0
            _, _, mask = self.samples[0]
            self.current_mask = mask.copy()

        return len(self.samples)

    def get_current_sample(self) -> Optional[tuple[str, np.ndarray, np.ndarray]]:
        """Get the current image sample as (name, image, mask)."""
        if not self.samples or not (0 <= self.current_index < len(self.samples)):
            return None
        return self.samples[self.current_index]

    def overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create overlay of mask on image with colored regions."""
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image.copy()

        mask_colour = cv2.applyColorMap((mask * 40).astype(np.uint8), cv2.COLORMAP_JET)
        mask_colour = cv2.cvtColor(mask_colour, cv2.COLOR_BGR2RGB)

        mask_binary = (mask > 0).astype(np.uint8)
        mask_binary_3ch = np.stack([mask_binary] * 3, axis=-1) * 255

        overlayed = np.where(
            mask_binary_3ch,
            cv2.addWeighted(img_rgb, 0.5, mask_colour, 0.5, 0),
            img_rgb,
        )
        return overlayed.astype(np.uint8)

    def array_to_base64(self, arr: np.ndarray) -> str:
        """Convert numpy array to base64 encoded PNG."""
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def navigate(self, delta: int):
        """Navigate to next/previous image."""
        new_index = self.current_index + delta
        if 0 <= new_index < len(self.samples):
            self.current_index = new_index
            _, _, mask = self.samples[self.current_index]
            self.current_mask = mask.copy()
            self.last_click = None
            self.mask_history = []  # Clear history when switching images
            self.refresh_display()

    def get_mask_id_at(self, x: int, y: int) -> int | None:
        """Get the mask ID at the given position."""
        if self.current_mask is None:
            return None
        if 0 <= y < self.current_mask.shape[0] and 0 <= x < self.current_mask.shape[1]:
            mask_id = int(self.current_mask[y, x])
            return mask_id if mask_id > 0 else None
        return None

    def save_mask_state(self):
        """Save the current mask state to history."""
        if self.current_mask is not None:
            self.mask_history.append(self.current_mask.copy())

    def undo(self):
        """Restore the previous mask state from history."""
        if self.mask_history:
            self.current_mask = self.mask_history.pop()
        else:
            ui.notify("Original Mask", type="warning")

    def handle_click(self, e: MouseEventArguments):
        """Handle mouse click on the image."""
        x, y = int(e.image_x), int(e.image_y)
        self.last_click = (x, y)
        mask_id = self.get_mask_id_at(x, y)
        # Save state before modification
        self.save_mask_state()
        # Check for Ctrl+click, mask exists and the click is on a mask.
        if e.ctrl and self.current_mask is not None and mask_id:
            self.current_mask = remove_pixel(self.current_mask, x, y)
        # Add functionality for Shift+Click to grow the selected mask
        elif e.shift and self.current_mask is not None and mask_id:
            self.grow_mask_action()
        # Add functionality for Alt+Click to shrink the selected mask
        elif e.alt and self.current_mask is not None and mask_id:
            self.shrink_mask_action()
        # If we have a normal click we colour the pixel to the nearest colour.
        elif not e.ctrl and self.current_mask is not None:
            self.current_mask = pixel_colour(self.current_mask, x, y)
        
        # No matter what if a click happened we refresh the image display.
        self.refresh_display()
        msg = f"(X, Y) = ({x}, {y}) | Mask ID: background"
        self.status_label.set_text(msg)

    def grow_mask_action(self):
        """Grow the selected mask."""
        if self.last_click is None:
            ui.notify("Click on a mask first", type="warning")
            return
        x, y = self.last_click
        mask_id = self.get_mask_id_at(x, y)
        if self.current_mask is not None and mask_id:
            self.save_mask_state()
            self.current_mask = grow_mask(self.current_mask.copy(), mask_id)

    def shrink_mask_action(self):
        """Shrink the selected mask."""
        if self.last_click is None:
            ui.notify("Click on a mask first", type="warning")
            return
        x, y = self.last_click
        mask_id = self.get_mask_id_at(x, y)
        if self.current_mask is not None and mask_id:
            self.save_mask_state()
            self.current_mask = shrink_mask(self.current_mask.copy(), mask_id)

    def merge_masks_action(self):
        """Merge the two closest masks."""
        if self.current_mask is not None:
            self.save_mask_state()
            self.current_mask = merge_mask(self.current_mask.copy())

    def split_connected_pixels(self):
        """Split connected pixels of selected mask into a new mask ID."""
        if self.last_click is None:
            ui.notify("Click on a mask first", type="warning")
            return
        
        if self.current_mask is None:
            ui.notify("No mask available", type="warning")
            return
        
        self.save_mask_state()
        x, y = self.last_click
        self.current_mask = split_connected(self.current_mask, x, y)

    def save_and_next(self):
        """Save current image and proceed to next."""
        # Only create saver on first save
        if self.saver is None:
            self.saver = DatasetSaver(str(self.output_path))
        
        # Ensure we get a none None sample
        sample = self.get_current_sample()
        if sample is None:
            return
        
        # Unpack the sample
        name, image, _ = sample
        
        # Use the current edited mask, not the original
        mask = self.current_mask if self.current_mask is not None else sample[2]
        
        # Check if already saved
        is_already_saved = self.saver.is_processed(original_name=name)
        
        if is_already_saved:
            # Delete the old version
            self.saver.delete_sample_by_original_name(original_name=name)
            ui.notify(f"Replaced {name}", type="info")
        
        # Save the (new or updated) sample
        self.saver.save_sample(image=image, mask=mask, original_name=name)
        
        if not is_already_saved:
            ui.notify(f"Saved {name}", type="positive")
        
        self.navigate(1)

    def refresh_display(self):
        """Refresh the image display."""
        sample = self.get_current_sample()
        if not sample:
            return

        name, img, original_mask = sample
        mask = self.current_mask if self.current_mask is not None else original_mask
        overlay_img = self.overlay(img, mask)
        
        # Update the interactive image source
        b64 = self.array_to_base64(overlay_img)
        self.interactive_image.set_source(f"data:image/png;base64,{b64}")
        
        # Update navigation info
        self.nav_label.set_text(f"Image {self.current_index + 1} of {len(self.samples)}: {name}")

