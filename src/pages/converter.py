"""Converter page — a page dedicated to converting .ome.tif directories to png images.."""

from nicegui import ui
from nicegui.elements.label import Label

from src.components.list_dir import ListDir
from src.components.local_dir_picker import LocalDirectoryPicker
from src.layout import page_layout


async def convert_images():
    pass


@ui.page("/converter")
def converter_page():
    with page_layout("Data Converter"):
        main_container = ui.column(wrap=False).classes("w-full h-full border")
        with main_container:
            with ui.column().classes("w-full border justify-center items-center"):
                ui.button(icon="swap_horizontal_circle", text="Convert").classes(
                    "text-xl icon-xl hover:animate-pulse"
                )
                ui.label("Progress Bar Here")
            with ui.column().classes("w-full border items-center"):
                ui.label("Data colums here")
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

                        with ui.column().classes("w-full items-center justify-center"):
                            with ui.row().classes("gap-2 justify-center items-center"):
                                ui.button(
                                    text="Input Path",
                                    icon="folder",
                                    on_click=lambda label: pick_data_dir(
                                        data_input_label, input_table
                                    ),
                                ).props("flat dense").tooltip("Browse…")
                                data_input_label = ui.label(
                                    "No directory selected"
                                ).classes("text-m text-gray-500")
                                # List a directory as a table, initially empty
                                input_table = ListDir()
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
                                # List a directory as a table, initially empty
                                output_table = ListDir()
