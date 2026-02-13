from nicegui import ui
from toolbox.ome_tif.legacy_viewer import LegacyViewer
from toolbox.mask_utils import array_to_base64, overlay

viewer = LegacyViewer('data')
image_channels = viewer.get_image_channels()

PIXELATED_STYLE = 'image-rendering: pixelated; max-height: 70vh;'


def get_next_batch():
    """Get the next list of (filename, image_array, mask_array, metadata) tuples."""
    try:
        return next(image_channels)
    except StopIteration:
        return None


@ui.page('/')
def main():
    # State: current batch data and overlay toggle
    state = {'batch': None, 'show_overlay': False}

    def render_batch():
        """Render the current batch, respecting the overlay toggle."""
        grid_container.clear()
        batch = state['batch']
        if batch is None:
            with grid_container:
                ui.label('No more images.').classes('text-h5')
            return
        with grid_container:
            with ui.grid(columns=len(batch)).classes('w-full gap-4 p-4'):
                for filename, image_array, mask_array, metadata in batch:
                    with ui.card().tight():
                        ui.label(filename).classes('p-2 font-bold')
                        if state['show_overlay'] and mask_array is not None:
                            display = overlay(image_array, mask_array)
                        else:
                            display = image_array
                        b64 = array_to_base64(display)
                        ui.image(f'data:image/png;base64,{b64}').style(PIXELATED_STYLE)
                        with ui.expansion('Metadata').classes('w-full'):
                            ui.json_editor({'content': {'json': metadata}})

    def show_next():
        state['batch'] = get_next_batch()
        render_batch()

    def toggle_overlay():
        state['show_overlay'] = not state['show_overlay']
        overlay_btn.props(f'color={"primary" if state["show_overlay"] else "grey"}')
        render_batch()

    with ui.row().classes('m-4 items-center gap-2'):
        ui.button('Next Image', on_click=show_next)
        overlay_btn = ui.button('Toggle Mask Overlay', on_click=toggle_overlay, color='grey')
    grid_container = ui.element('div').classes('w-full')

    show_next()


ui.run(port=8080, title='OME TIFF Viewer', reload=True, dark=True)