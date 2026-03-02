"""OME TIFF Viewer page — browse OME-TIFF images with optional mask overlay."""

import os

from nicegui import ui
from nicegui.elements.label import Label

from src.components.local_picker import LocalDirectoryPicker
from src.layout import page_layout
from toolbox.ome_converter import OMEConverter
from toolbox.utilities import array_to_base64, overlay

DATA_ROOT = os.getenv("DATA_ROOT", "/data")
PIXELATED_STYLE = "image-rendering: pixelated; max-height: 70vh;"
PAGE_HELP = """
The OME TIF viewer can be used to view ome.tif images with their respective default masks.

- Set the input path. The target directory should contain an Exported Masks folder with all the .ome.tif files.
- Cycle through the images using Previous / Next and view.

"""


@ui.page("/viewer")
def viewer_page():
    state: dict = {
        "show_overlay": False,
        "converter": None,
        "index": 0,
    }

    def on_slider_change(e):
        state["index"] = int(e.value)
        render()

    def render():
        """Load and render channels for the current index."""
        grid_container.clear()
        converter: OMEConverter | None = state["converter"]
        if converter is None or not converter.filenames:
            with grid_container:
                ui.label("No images loaded.").classes("text-h5")
            return

        idx = state["index"]
        slider.set_value(idx)
        channels = converter.load_channels(idx)
        if not channels:
            with grid_container:
                ui.label("Could not load image.").classes("text-h5")
            return

        with grid_container:
            ui.label(f"Image {idx + 1} / {len(converter.filenames)}").classes(
                "text-subtitle1 p-2"
            )
            with ui.grid(columns=len(channels)).classes("w-full gap-4 p-4"):
                for filename, image_array, mask_array, metadata in channels:
                    with ui.card().tight():
                        ui.label(filename).classes("p-2 font-bold")
                        if state["show_overlay"] and mask_array is not None:
                            display = overlay(image_array, mask_array)
                        else:
                            display = image_array
                        b64 = array_to_base64(display)
                        ui.image(f"data:image/png;base64,{b64}").style(PIXELATED_STYLE)
                        with ui.expansion("Metadata").classes("w-full"):
                            ui.json_editor({"content": {"json": metadata}})

    def show_next():
        converter: OMEConverter | None = state["converter"]
        if converter and state["index"] < len(converter.filenames) - 1:
            state["index"] += 1
            render()

    def show_prev():
        if state["index"] > 0:
            state["index"] -= 1
            render()

    def toggle_overlay():
        state["show_overlay"] = not state["show_overlay"]
        overlay_btn.props(f"color={'primary' if state['show_overlay'] else 'grey'}")
        render()

    with page_layout("OME TIFF Viewer", PAGE_HELP):

        async def pick_data_dir(label: Label):
            result = await LocalDirectoryPicker(
                directory=label.text or DATA_ROOT,
                title="Select Data Directory",
            )
            if result:
                label.set_text(result)
                data_path = label.text.strip()
                try:
                    state["converter"] = OMEConverter(input_path=data_path)
                    state["index"] = 0
                    slider._props["max"] = max(len(state["converter"].filenames) - 1, 0)
                    slider.set_value(0)
                    slider.update()
                    render()
                    ui.notify(f"Loaded data from {data_path}", type="positive")
                except Exception as exc:
                    ui.notify(f"Error loading data: {exc}", type="negative")

        with ui.column().classes("m-4 items-center gap-2 w-full"):
            with ui.row():
                ui.button(
                    text="Input Path",
                    icon="folder",
                    on_click=lambda label: pick_data_dir(data_input_label),
                ).props("flat dense").tooltip("Select Input")
                data_input_label = ui.label("No directory selected").classes(
                    "text-m text-gray-500"
                )
            with ui.row():
                ui.button("Previous", icon="arrow_back", on_click=show_prev)
                ui.button("Next", icon="arrow_forward", on_click=show_next)
                overlay_btn = ui.button(
                    "Toggle Mask Overlay", on_click=toggle_overlay, color="grey"
                )
            slider = (
                ui.slider(min=0, max=0, step=1, on_change=on_slider_change)
                .props("label-always")
                .classes("w-full max-w-lg")
            )

        grid_container = ui.element("div").classes("w-full")
