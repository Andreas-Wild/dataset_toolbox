"""OME TIFF Viewer page — browse legacy OME-TIFF images with optional mask overlay."""

from nicegui import ui

from toolbox.ome_tif.legacy_viewer import LegacyViewer
from toolbox.mask_utils import array_to_base64, overlay
from src.layout import page_layout
from src.components.local_dir_picker import LocalDirectoryPicker

PIXELATED_STYLE = "image-rendering: pixelated; max-height: 70vh;"


@ui.page("/viewer")
def viewer_page():
    # Per-session state — the viewer and generator are created when the user
    # provides a data path and clicks "Load".
    state: dict = {
        "batch": None,
        "show_overlay": False,
        "viewer": None,
        "image_channels": None,
    }

    def _get_next_batch():
        """Get the next list of (filename, image_array, mask_array, metadata) tuples."""
        if state["image_channels"] is None:
            return None
        try:
            return next(state["image_channels"])
        except StopIteration:
            return None

    def render_batch():
        """Render the current batch, respecting the overlay toggle."""
        grid_container.clear()
        batch = state["batch"]
        if batch is None:
            with grid_container:
                ui.label("No more images.").classes("text-h5")
            return
        with grid_container:
            with ui.grid(columns=len(batch)).classes("w-full gap-4 p-4"):
                for filename, image_array, mask_array, metadata in batch:
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

    def load_viewer():
        """Instantiate the viewer from the provided data path."""
        data_path = data_input.value.strip()
        if not data_path:
            ui.notify("Please enter a data path", type="warning")
            return
        try:
            state["viewer"] = LegacyViewer(data_path)
            state["image_channels"] = state["viewer"].get_image_channels()
            show_next()
            ui.notify(f"Loaded data from {data_path}", type="positive")
        except Exception as exc:
            ui.notify(f"Error loading data: {exc}", type="negative")

    def show_next():
        state["batch"] = _get_next_batch()
        render_batch()

    def toggle_overlay():
        state["show_overlay"] = not state["show_overlay"]
        overlay_btn.props(f'color={"primary" if state["show_overlay"] else "grey"}')
        render_batch()

    with page_layout("OME TIFF Viewer"):
        async def pick_data_dir():
            result = await LocalDirectoryPicker(
                directory=data_input.value or "~",
                title="Select Data Directory",
            )
            if result:
                data_input.value = result

        with ui.row().classes("m-4 items-center gap-2 w-full"):
            data_input = ui.input("Data Path", value="data").classes("w-64")
            ui.button(icon="folder", on_click=pick_data_dir).props(
                "flat dense"
            ).tooltip("Browse…")
            ui.button("Load", on_click=load_viewer, icon="folder_open")
            ui.button("Next Image", on_click=show_next)
            overlay_btn = ui.button(
                "Toggle Mask Overlay", on_click=toggle_overlay, color="grey"
            )

        grid_container = ui.element("div").classes("w-full")
