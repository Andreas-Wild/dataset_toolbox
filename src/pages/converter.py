"""Converter page — a page dedicated to converting .ome.tif directories to png images.."""

import asyncio

from nicegui import ui
from nicegui.elements.label import Label

from src.components.list_dir import ListDir
from src.components.local_dir_picker import LocalDirectoryPicker
from src.layout import page_layout
from toolbox.data_converter import OMEConverter

PAGE_HELP = """
The Data Converter may be used to convert files from `.ome.tif` to `png`. Set the input directory and output directory by clicking on the respective '🗀' icon.

- **It is the responsibility of the user to select the correct output directory! No additional warnings will be given**.
- Ensure that the input directory contains `.ome.tif` files and optionally an ExportedMasks directory.
- If files with the same name already exist in the output dir, they will be overwritten.
"""


@ui.page("/converter")
def converter_page():
    with page_layout("Data Converter", help_text=PAGE_HELP):
        main_container = ui.column(wrap=False).classes("w-full h-full")
        with main_container:
            with ui.column().classes("w-full justify-center items-center"):

                async def run_conversion():
                    input_dir = data_input_label.text
                    output_dir = data_output_label.text
                    if (
                        input_dir == "No directory selected"
                        or output_dir == "No directory selected"
                    ):
                        ui.notify(
                            "Please select both input and output directories.",
                            type="warning",
                        )
                        return
                    if progress_bar.value != 0:
                        ui.notify(
                            "This conversion has already been completed. Please choose another directory"
                        )
                        return
                    convert_btn.disable()
                    progress_bar.visible = True
                    status_label.set_text("Converting...")
                    progress_bar.set_value(0)

                    try:
                        converter = OMEConverter(input_dir, output_dir)
                        await asyncio.get_event_loop().run_in_executor(
                            None, converter.convert
                        )
                        status_label.set_text("Conversion complete!")
                        progress_bar.set_value(1)
                        ui.notify("Conversion complete!", type="positive")
                        output_table.load(output_dir)
                    except Exception as e:
                        status_label.set_text(f"Error: {e}")
                        ui.notify(f"Conversion failed: {e}", type="negative")
                    finally:
                        convert_btn.enable()
                        status_label.set_text("")

                convert_btn = ui.button(
                    icon="swap_horizontal_circle",
                    text="Convert",
                    on_click=run_conversion,
                ).classes("text-xl icon-xl hover:animate-pulse")

                progress_bar = ui.linear_progress(value=0, show_value=False).classes(
                    "w-3/4"
                )
                progress_bar.visible = False
                status_label = ui.label("").classes("text-sm text-gray-400")

            with ui.column().classes("w-full items-center"):
                with ui.splitter(value=50, limits=(25, 75)).classes(
                    "w-full"
                ) as splitter:
                    with splitter.before:
                        # Here we need to create a callback to pick directories.
                        async def pick_data_dir(label: Label, table: ListDir):
                            result = await LocalDirectoryPicker(
                                directory=label.text or "~",
                                title="Select Data Directory",
                            )
                            if result:
                                label.set_text(result)
                                table.load(result)
                                # Also we want to reset the progress bar to indicate that we have more to convert
                                progress_bar.set_value(0)

                        with ui.column().classes("w-full items-center justify-center"):
                            with ui.row().classes("gap-2 justify-center items-center"):
                                ui.button(
                                    text="Input Path",
                                    icon="folder",
                                    on_click=lambda label: pick_data_dir(
                                        data_input_label, input_table
                                    ),
                                ).props("flat dense").tooltip("Select Input")
                                data_input_label = ui.label(
                                    "No directory selected"
                                ).classes("text-m text-gray-500")
                                # List a directory as a table, initially empty
                                input_table = ListDir()
                                input_table.set_visibility(False)
                    with splitter.after:
                        with ui.column().classes("w-full items-center justify-center"):
                            with ui.row().classes("gap-2 justify-center items-center"):
                                ui.button(
                                    text="Output Path",
                                    icon="folder",
                                    on_click=lambda label: pick_data_dir(
                                        data_output_label, output_table
                                    ),
                                ).props("flat dense").tooltip("Browse…")
                                data_output_label = ui.label(
                                    "No directory selected"
                                ).classes("text-m text-gray-500")
                                # List a directory as a table, initially empty and invisible
                                output_table = ListDir()
                                output_table.set_visibility(False)
