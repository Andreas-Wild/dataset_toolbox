"""Dataset Editor page — interactive mask editing for image datasets."""

from pathlib import Path

import numpy as np
from nicegui import ui
from nicegui.elements.label import Label
from nicegui.events import KeyEventArguments

from src.components.data_editor import DatasetEditor
from src.components.local_dir_picker import LocalDirectoryPicker
from src.layout import page_layout
from toolbox.utilities import array_to_base64

PAGE_HELP = """
The Image Dataset Editor page can be used to edit masks at the pixel level. This allows the user to have very fine-grained control over the final masks thee use for training.
There are a wide array of tools that the user may use to edit the masks.

**Getting started**:

- Set the 'INPUT PATH' to the masks you want to edit. This folder should contain:
    1. An `image_source.txt` file that points to the image source.
    2. A `masks` folder with png images of masks.
    3. A `JSON` file containing all the label studio annotations
- If your folder does not yet look like this use the RLE Converter tool **first**.
- A suggested 'OUTPUT PATH' will automatically be set. Be sure to double check this is the correct location.
"""


@ui.page("/editor")
def editor_page():
    editor = DatasetEditor()

    with page_layout("Image Dataset Editor", help_text=PAGE_HELP):
        with ui.splitter(value=20, limits=(0, 30)).classes("w-full h-full") as splitter:
            # Sidebar
            with splitter.before:
                with ui.column().classes("p-4 gap-4 w-full"):
                    ui.label("Dataset Paths").classes("text-lg font-semibold")

                    async def pick_data_dir(label: Label):
                        result: str = await LocalDirectoryPicker(
                            directory=label.text or "~",
                            title="Select Data Directory",
                        )
                        if result:
                            label.set_text(result)

                        # Set the output path to the same input path but replace 'annotations' with 'completed' if there is no output yet
                        if data_output_label.text == "No directory selected":
                            data_output_label.set_text(
                                result.replace("annotations", "complete")
                            )

                    with ui.row().classes("w-full items-end gap-1"):
                        ui.button(
                            text="Input Path",
                            icon="folder",
                            on_click=lambda label: pick_data_dir(data_input_label),
                        ).props("flat dense").tooltip("Select Input")
                        data_input_label = ui.label("No directory selected").classes(
                            "text-m text-gray-500"
                        )

                    with ui.row().classes("w-full items-end gap-1"):
                        ui.button(
                            text="Ouput Path",
                            icon="folder",
                            on_click=lambda label: pick_data_dir(data_output_label),
                        ).props("flat dense").tooltip("Select Output")
                        data_output_label = ui.label("No directory selected").classes(
                            "text-m text-gray-500"
                        )

                    def load_dataset():
                        editor.dataset_path = Path(data_input_label.text)
                        editor.output_path = Path(data_output_label.text)
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
                        ui.button(
                            "◀ Prev", on_click=lambda: editor.navigate(-1)
                        ).classes("flex-1")
                        ui.button(
                            "Next ▶", on_click=lambda: editor.navigate(1)
                        ).classes("flex-1")

                    ui.separator()

                    # Controls help
                    ui.label("Controls").classes("text-lg font-semibold")
                    ui.markdown("""
                    - **`'Left'`** and **`'Right'`** arrows to navigate between images
                    - Save images to output path with **`'Enter'`**
                    - **`'Click'`** to colour a pixel & select a mask
                    - **`'Ctrl+Click'`** to remove colour from a pixel
                    - Use **`'Backspace'`** to undo
                    - Use **`'g'`** or **`'Shift+Click'`** to grow the selected mask
                    - Use **`'s'`** or **`'Alt+Click'`** to shrink the selected mask
                    - Use **`'m'`** to merge the two closest masks
                    - Use **`'c'`** to split connected pixels to new mask ID
                    """)

            # Main content area
            with splitter.after:
                with ui.column().classes(
                    "w-full h-full items-center justify-center p-4"
                ):
                    placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
                    b64 = array_to_base64(placeholder)

                    editor.interactive_image = ui.interactive_image(
                        source=f"data:image/png;base64,{b64}",
                        cross=False,
                        on_mouse=editor.handle_click,
                        events=["mousedown"],
                    ).style(
                        "image-rendering: pixelated; min-height: 80vh; width: auto; "
                        "user-select: none; -webkit-user-select: none;"
                    )

    # Keyboard shortcuts
    ui.keyboard(
        on_key=lambda e: _handle_keyboard(e, editor),
        ignore=["input", "textarea"],
    )


def _handle_keyboard(e: KeyEventArguments, editor: DatasetEditor):
    """Handle keyboard shortcuts for the editor page."""
    if e.action.keydown and not e.action.repeat:
        if e.key.arrow_left:
            editor.navigate(-1)
        elif e.key.arrow_right:
            editor.navigate(1)
        elif e.key.enter:
            editor.save_and_next()
        elif e.key.backspace:
            editor.undo()
        elif e.key.name == "g":
            editor.grow_mask_action()
        elif e.key.name == "s":
            editor.shrink_mask_action()
        elif e.key.name == "m":
            editor.merge_masks_action()
        elif e.key.name == "c":
            editor.split_connected_pixels()

        editor.refresh_display()
