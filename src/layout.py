"""Shared layout components for the Dataset Toolbox app."""

from contextlib import contextmanager
from nicegui import ui


@contextmanager
def page_layout(title: str):
    """Wrap page content with a shared header and navigation."""
    ui.dark_mode().enable()

    with ui.header().classes("items-center justify-between"):
        ui.label("Dataset Toolbox").classes("text-2xl font-bold")
        with ui.row().classes("gap-2"):
            ui.button("Home", on_click=lambda: ui.navigate.to("/"), icon="home").props("flat color=white")
            ui.button("Editor", on_click=lambda: ui.navigate.to("/editor"), icon="edit").props("flat color=white")
            ui.button("Viewer", on_click=lambda: ui.navigate.to("/viewer"), icon="visibility").props("flat color=white")

    # Yield control so the caller can add page-specific content
    yield
