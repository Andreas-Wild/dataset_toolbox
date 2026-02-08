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
    grid_container = ui.element('div').classes('w-full')

    def show_next():
        grid_container.clear()
        batch = get_next_batch()
        if batch is None:
            with grid_container:
                ui.label('No more images.').classes('text-h5')
            return
        with grid_container:
            with ui.grid(columns=len(batch)).classes('w-full gap-4 p-4'):
                for filename, image_array, mask_array, metadata in batch:
                    with ui.card().tight():
                        ui.label(filename).classes('p-2 font-bold')
                        b64 = array_to_base64(image_array)
                        ui.image(f'data:image/png;base64,{b64}').style(PIXELATED_STYLE)
                        if mask_array is not None:
                            b64_mask = array_to_base64(mask_array)
                            ui.label('Mask:').classes('p-2')
                            ui.image(f'data:image/png;base64,{b64_mask}').style(PIXELATED_STYLE)
                        with ui.expansion('Metadata').classes('w-full'):
                            ui.json_editor({'content': {'json': metadata}})

    ui.button('Next Batch', on_click=show_next).classes('m-4')
    show_next()


ui.run(port=8080, title='OME TIFF Viewer', reload=True, dark=True)