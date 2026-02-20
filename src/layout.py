"""Shared layout components for the Dataset Toolbox app."""

from contextlib import contextmanager

from nicegui import app, ui


@contextmanager
def page_layout(title: str, help_text: str | None = None):
    """Wrap page content with a shared header and navigation.

    Args:
        title: Page title shown in the header.
        help_text: Optional markdown string shown in a help dialog when the
                   help button is clicked.  If omitted, the button is hidden.
    """
    ui.dark_mode().enable()

    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("gap-2"):
            ui.label(title).classes("text-2xl font-bold")

            if help_text is not None:
                storage_key = f"help_seen_{title}"

                with ui.dialog() as help_dialog, ui.card().classes("max-w-lg"):
                    ui.label(title).classes("text-xl font-bold")
                    ui.separator()
                    ui.markdown(help_text)
                    ui.button("Close", on_click=help_dialog.close).props("flat")

                ui.button(
                    icon="help",
                    color=None,
                    on_click=help_dialog.open,
                ).props("flat color=white").tooltip("Help")

                def _auto_open(key=storage_key):
                    if not app.storage.user.get(key, False):
                        app.storage.user[key] = True
                        help_dialog.open()

                ui.timer(0, _auto_open, once=True)

        with ui.row().classes("gap-2"):
            ui.button(
                "Editor", on_click=lambda: ui.navigate.to("/editor"), icon="edit"
            ).props("flat color=white")
            ui.button(
                "Viewer", on_click=lambda: ui.navigate.to("/viewer"), icon="visibility"
            ).props("flat color=white")
            ui.button(
                "Converter",
                on_click=lambda: ui.navigate.to("/converter"),
                icon="swap_horizontal_circle",
            ).props("flat color=white")
            ui.button("Home", on_click=lambda: ui.navigate.to("/"), icon="home").props(
                "flat color=white"
            )

    # Yield control so the caller can add page-specific content
    yield
