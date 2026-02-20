"""Converter page — a page dedicated to converting .ome.tif directories to png images.."""
from nicegui.elements.input import Input
from pathlib import Path

from nicegui import ui

from src.layout import page_layout
from src.components.local_dir_picker import LocalDirectoryPicker


@ui.page("/converter")
def converter_page():
    with page_layout("Data Converter"):
        async def pick_data_dir():
            result = await LocalDirectoryPicker(
                directory=data_input.value or "~",
                title="Select Data Directory",
            )
            if result:
                data_input.value = result
        with ui.column().classes("w-full items-center justify-center p-8 gap-8"):
            ui.label("Dataset Converter").classes("text-4xl font-bold")

            with ui.row().classes("m-4 items-center gap-2 w-full"):
                data_input: Input = ui.input("Data Path", placeholder="Click the 📁 to pick a directory.").classes("w-64")
                ui.button(icon="📁", on_click=pick_data_dir).props(
                    "flat dense"
                ).tooltip("Browse…")
            with ui.row().classes("m-4 items-center gap-2 w-full"):
                ls_dir = ui.markdown("")
                ls_dir.set_visibility(False)
                
            def on_path_change(e):
                data_path = Path(e.value)
                if data_path.is_dir():
                    num_tif_files = len([f for f in data_path.rglob("*.ome.tif")])
                    num_mask_files = len([f for f in data_path.rglob("*.dmask.pgm")])
                    ls_dir.set_content(f"""
                    Directory {data_input.value}: \n
                        - Found {num_tif_files} .ome.tif files \n
                        - Found {num_mask_files} .dmask.pgm files
                    """)
                    ls_dir.set_visibility(True)
                else:
                    ls_dir.set_visibility(False)


        data_input.on_value_change(on_path_change)

