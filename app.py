"""Minimal NiceGUI port of the Image Dataset Editor."""

import base64
import io
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from nicegui import ui


class DatasetEditor:
    def __init__(self):
        self.dataset_path: Path = Path.cwd() / "input"
        self.output_path: Path = Path.cwd() / "output"
        self.samples: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.current_index: int = 0
        self.current_mask: Optional[np.ndarray] = None
        self.last_click: Optional[tuple[int, int]] = None
        self.interactive_image: ui.interactive_image = ui.interactive_image()
        self.status_label: ui.label = ui.label()
        self.nav_label: ui.label = ui.label()

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
        """Get the current image sample."""
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
            self.refresh_display()

    def handle_click(self, e):
        """Handle mouse click on the image."""
        x, y = int(e.image_x), int(e.image_y)
        self.last_click = (x, y)
        
        mask_id = None
        if self.current_mask is not None:
            if 0 <= y < self.current_mask.shape[0] and 0 <= x < self.current_mask.shape[1]:
                mask_id = int(self.current_mask[y, x])
                if mask_id == 0:
                    mask_id = None
        
        msg = f"Clicked: ({x}, {y}) | Mask ID: {mask_id if mask_id else 'background'}"
        self.status_label.set_text(msg)

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




@ui.page("/")
def main_page():
    # Create the editor instance
    editor = DatasetEditor()
    ui.dark_mode().enable()

    with ui.header().classes("items-center justify-between"):
        ui.label("Image Dataset Editor").classes("text-2xl font-bold")

    with ui.splitter(value=20, limits=(15,40)).classes("w-full h-full") as splitter:
        # Sidebar
        with splitter.before:
            with ui.column().classes("p-4 gap-4 w-full"):
                ui.label("Dataset Paths").classes("text-lg font-semibold")

                input_path = ui.input(
                    "Input Path",
                    value=str(editor.dataset_path),
                ).classes("w-full")

                output_path = ui.input(
                    "Output Path", 
                    value=str(editor.output_path),
                ).classes("w-full")

                def load_dataset():
                    editor.dataset_path = Path(input_path.value)
                    editor.output_path = Path(output_path.value)
                    count = editor.load_dataset()
                    if count > 0:
                        ui.notify(f"Loaded {count} images", type="positive")
                        editor.refresh_display()
                    else:
                        ui.notify("No images found", type="warning")

                ui.button("Load Dataset", on_click=load_dataset).classes("w-full")

                ui.separator()

                # Navigation
                ui.label("Navigation").classes("text-lg font-semibold")
                editor.nav_label = ui.label("No dataset loaded")

                with ui.row().classes("w-full gap-2"):
                    ui.button("◀ Prev", on_click=lambda: editor.navigate(-1)).classes("flex-1")
                    ui.button("Next ▶", on_click=lambda: editor.navigate(1)).classes("flex-1")

                ui.separator()

                # Status
                ui.label("Click Info").classes("text-lg font-semibold")
                editor.status_label = ui.label("Click on image to see coordinates")

        # Main content area
        with splitter.after:
            with ui.column().classes("w-full h-full items-center justify-center p-4"):
                # Create a placeholder image
                placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
                b64 = editor.array_to_base64(placeholder)
                
                editor.interactive_image = ui.interactive_image(
                    source=f"data:image/png;base64,{b64}",
                    cross=False,
                    on_mouse=editor.handle_click,
                    events=["mousedown"],
                ).style("image-rendering: pixelated; min-height: 80vh; width: auto;")


ui.run(title="Image Dataset Editor", port=8080, reload=True)
