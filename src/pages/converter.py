"""Converter page — a page dedicated to converting .ome.tif directories to png images.."""

from nicegui import ui

from src.layout import page_layout


@ui.page("/converter")
def converter_page():
    with page_layout("Data Converter"):
        with ui.column().classes("w-full items-center justify-center p-8 gap-8"):
            ui.label("Dataset Converter").classes("text-4xl font-bold")

            ui.markdown(content="This is a blank page and still needs to be implemented.")
