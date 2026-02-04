from nicegui import ui
from toolbox.data_editor import DatasetEditor
from pathlib import Path
import numpy as np
from nicegui.events import KeyEventArguments


@ui.page("/")
def main_page():
    # Create the editor instance
    editor = DatasetEditor()
    ui.dark_mode().enable()

    with ui.header().classes("items-center justify-between"):
        ui.label("Image Dataset Editor").classes("text-2xl font-bold")

    with ui.splitter(value=20, limits=(0,30)).classes("w-full h-full") as splitter:
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
            with ui.column().classes("w-full h-full items-center justify-center p-4"):
                # Create a placeholder image
                placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
                b64 = editor.array_to_base64(placeholder)
                
                editor.interactive_image = ui.interactive_image(
                    source=f"data:image/png;base64,{b64}",
                    cross=False,
                    
                    on_mouse=editor.handle_click,
                    events=["mousedown"],
                ).style("image-rendering: pixelated; min-height: 80vh; width: auto; user-select: none; -webkit-user-select: none;")

    # Register keyboard shortcuts
    ui.keyboard(
        on_key=lambda e: handle_keyboard(e, editor),
        ignore=["input", "textarea"],
    )


def handle_keyboard(e: KeyEventArguments, editor: DatasetEditor):
    """Handle keyboard shortcuts."""
    if e.action.keydown and not e.action.repeat:
        # Arrow keys for navigation
        if e.key.arrow_left:
            editor.navigate(-1)
        elif e.key.arrow_right:
            editor.navigate(1)
        
        # Tab to save and proceed
        elif e.key.enter:
            editor.save_and_next()
        
        # Backspace to undo
        elif e.key.backspace:
            editor.undo()
        
        # G to grow mask
        elif e.key.name == "g":
            editor.grow_mask_action()
        
        # S to shrink mask  
        elif e.key.name == "s":
            editor.shrink_mask_action()
        
        # M to merge masks
        elif e.key.name == "m":
            editor.merge_masks_action()
        
        # C to split connected pixels
        elif e.key.name == "c":
            editor.split_connected_pixels()

        editor.refresh_display()


ui.run(title="Image Dataset Editor", port=8080, reload=True)