"""Home page — landing page with navigation to all tools."""

from nicegui import ui

from src.layout import page_layout


@ui.page("/")
def home_page():
    with page_layout("Home"):
        with ui.column().classes("w-full items-center justify-center p-8 gap-8"):
            ui.label("Dataset Toolbox").classes("text-4xl font-bold")
            ui.label("Select a tool to get started.").classes("text-lg text-gray-400")

            with ui.row().classes("gap-6"):
                with ui.card().on("click", lambda: ui.navigate.to("/editor")).classes(
                    "cursor-pointer hover:shadow-lg"
                ):
                    with ui.column().classes("items-center p-4 gap-2"):
                        ui.icon("edit", size="xl")
                        ui.label("Dataset Editor").classes("text-lg font-semibold")
                        ui.label("Interactively edit masks for image datasets.").classes(
                            "text-sm text-gray-400 text-center"
                        )

                with ui.card().on("click", lambda: ui.navigate.to("/viewer")).classes(
                    "cursor-pointer hover:shadow-lg"
                ):
                    with ui.column().classes("items-center p-4 gap-2"):
                        ui.icon("visibility", size="xl")
                        ui.label("OME TIFF Viewer").classes("text-lg font-semibold")
                        ui.label("Browse legacy OME-TIFF images with mask overlays.").classes(
                            "text-sm text-gray-400 text-center"
                        )
                with ui.card().on("click", lambda: ui.navigate.to("/converter")).classes(
                    "cursor-pointer hover:shadow-lg"
                ):
                    with ui.column().classes("items-center p-4 gap-2"):
                        ui.icon("visibility", size="xl")
                        ui.label("Dataset Converter").classes("text-lg font-semibold")
                        ui.label("Convert `.ome.tif` file to png").classes(
                            "text-sm text-gray-400 text-center"
                        )
