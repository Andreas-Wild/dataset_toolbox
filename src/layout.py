"""Shared layout components for the Dataset Toolbox app."""

from contextlib import contextmanager

from nicegui import app, ui

from src.ml_backends import ML_BACKENDS, MLBackend


def _add_backend_chip(backend: MLBackend) -> None:
    """Add an icon-only status button for an ML backend with a detail dialog."""

    with ui.dialog() as detail_dialog, ui.card().classes("min-w-[350px]"):
        ui.label(backend.name).classes("text-xl font-bold")
        ui.separator()
        detail_container = ui.column().classes("gap-1 w-full")
        ui.button("Close", on_click=detail_dialog.close).props("flat")

    btn = ui.button(icon=backend.icon, on_click=detail_dialog.open).props(
        "flat round dense color=grey"
    )
    btn.tooltip("GPU")

    async def _update():
        health = await backend.fetch_health()
        detail_container.clear()

        if health is None:
            btn.props(remove="color")
            btn.props("color=red")
            with detail_container:
                _detail_row("Status", "Offline")
            return

        cuda = health.get("cuda", {})
        model = health.get("model", {})
        cuda_available = cuda.get("available", False)

        if cuda_available:
            btn.props(remove="color")
            btn.props("color=green")
        else:
            btn.props(remove="color")
            btn.props("color=orange")

        with detail_container:
            _detail_row("Status", "Online")
            _detail_row("Model", model.get("checkpoint", "?"))
            _detail_row("CUDA Version", cuda.get("version", "N/A"))
            _detail_row("PyTorch sees CUDA", str(cuda_available))
            _detail_row("Inference Device", health.get("device", "?"))
            _detail_row("NVIDIA Driver", cuda.get("nvidia_driver", "N/A"))
            devices = cuda.get("devices", [])
            if devices:
                names = ", ".join(d["name"] for d in devices)
                _detail_row("Available GPUs", names)
            else:
                _detail_row("Available GPUs", "None")

    ui.timer(0, _update, once=True)


def _detail_row(label: str, value: str) -> None:
    """Render a label: value row inside the detail dialog."""
    with ui.row().classes("justify-between w-full items-center"):
        ui.label(label).classes("text-bold text-grey-5")
        ui.label(value)


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
        with ui.row().classes("gap-2 items-center"):
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

            for backend in ML_BACKENDS:
                _add_backend_chip(backend)

        with ui.row().classes("gap-2"):
            ui.button(
                "Label Studio",
                on_click=lambda: ui.navigate.to("/labelstudio"),
                icon="label",
            ).props("flat color=white").tooltip("Open Label Studio for annotation.")
            ui.button(
                "Editor", on_click=lambda: ui.navigate.to("/editor"), icon="edit"
            ).props("flat color=white").tooltip("Edit masks with fine-grained control.")
            ui.button(
                "Viewer", on_click=lambda: ui.navigate.to("/viewer"), icon="visibility"
            ).props("flat color=white").tooltip(
                "View `.ome.tif` images with their default masks."
            )
            ui.button(
                "RLE Converter",
                on_click=lambda: ui.navigate.to("/rleconverter"),
                icon="change_circle",
            ).props("flat color=white").tooltip(
                "Convert Label Studio `RLE` masks to `png` image masks."
            )
            ui.button(
                "OME Converter",
                on_click=lambda: ui.navigate.to("/converter"),
                icon="swap_horizontal_circle",
            ).props("flat color=white").tooltip(
                "Convert `ome.tif` images to `png` and group them by channel."
            )
            ui.button("Home", on_click=lambda: ui.navigate.to("/"), icon="home").props(
                "flat color=white"
            ).tooltip("Return home")

    # Yield control so the caller can add page-specific content
    yield
