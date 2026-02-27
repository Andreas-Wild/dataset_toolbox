"""Reusable local filesystem picker dialogs for NiceGUI.

Provides two dialogs:
- LocalDirectoryPicker: browse and select a directory
- LocalFilePicker: browse and select a file, with optional file type filtering
"""

from pathlib import Path
from typing import List, Optional

from nicegui import ui


class _BaseLocalPicker(ui.dialog):
    """Shared navigation logic for directory and file pickers."""

    def __init__(
        self,
        directory: str = "~",
        *,
        title: str = "Select",
        show_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.path = Path(directory).expanduser().resolve()
        if not self.path.is_dir():
            self.path = Path.home()
        self.show_hidden = show_hidden

        with self, ui.card().classes("size-128"):
            ui.label(title).classes("text-lg font-semibold")
            self.path_display = ui.label(str(self.path)).classes(
                "text-l font-bold break-all"
            )
            self.entries_list = ui.scroll_area().classes("w-full border rounded")
            self.entries_list.style("height: 500px")

            with ui.row().classes("w-full justify-between gap-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                self.select_btn = ui.button("Select", on_click=self._select).props(
                    "color=primary"
                )

        self._update_list()
        self.open()

    def _navigate(self, path: Path) -> None:
        if path.is_dir():
            self.path = path
            self._update_list()

    def _update_list(self) -> None:
        raise NotImplementedError

    def _select(self) -> None:
        raise NotImplementedError


class LocalDirectoryPicker(_BaseLocalPicker):
    """A dialog that lets users navigate and pick a local directory."""

    def __init__(
        self,
        directory: str = "~",
        *,
        title: str = "Select Directory",
        show_hidden: bool = False,
    ) -> None:
        super().__init__(directory, title=title, show_hidden=show_hidden)

    def _update_list(self) -> None:
        self.entries_list.clear()
        with self.entries_list:
            parent = self.path.parent
            with ui.item(on_click=lambda p=parent: self._navigate(p)).classes("w-full"):
                with ui.item_section().props("avatar"):
                    ui.icon("folder", color="amber")
                with ui.item_section():
                    ui.item_label(text="⟵ Go Back")

            ui.separator()

            try:
                entries = sorted(self.path.iterdir(), key=lambda p: p.name.lower())
            except PermissionError:
                entries = []
            for entry in entries:
                if not entry.is_dir():
                    continue
                if not self.show_hidden and entry.name.startswith("."):
                    continue
                with ui.item(on_click=lambda p=entry: self._navigate(p)).classes(
                    "w-full"
                ):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder", color="amber")
                    with ui.item_section():
                        ui.item_label(entry.name)

        self.path_display.text = str(self.path)

    def _select(self) -> None:
        self.submit(str(self.path))


class LocalFilePicker(_BaseLocalPicker):
    """A dialog that lets users navigate directories and pick a file.

    Args:
        directory: Initial directory to open.
        allowed_types: List of allowed file extensions, e.g. ``[".csv", ".json"]``.
            Pass ``None`` (default) to show all files.
        title: Dialog title.
        show_hidden: Whether to show hidden files and directories.
    """

    def __init__(
        self,
        directory: str = "~",
        *,
        allowed_types: Optional[List[str]] = None,
        title: str = "Select File",
        show_hidden: bool = False,
    ) -> None:
        self.allowed_types = (
            [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_types]
            if allowed_types is not None
            else None
        )
        self.selected_file: Optional[Path] = None
        super().__init__(directory, title=title, show_hidden=show_hidden)
        self.select_btn.disable()

    def _update_list(self) -> None:
        self.entries_list.clear()
        self.selected_file = None
        if hasattr(self, "select_btn"):
            self.select_btn.disable()

        with self.entries_list:
            parent = self.path.parent
            with ui.item(on_click=lambda p=parent: self._navigate(p)).classes("w-full"):
                with ui.item_section().props("avatar"):
                    ui.icon("folder", color="amber")
                with ui.item_section():
                    ui.item_label(text="⟵ Go Back")

            ui.separator()

            try:
                entries = sorted(self.path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                entries = []

            for entry in entries:
                if not self.show_hidden and entry.name.startswith("."):
                    continue

                if entry.is_dir():
                    with ui.item(on_click=lambda p=entry: self._navigate(p)).classes(
                        "w-full"
                    ):
                        with ui.item_section().props("avatar"):
                            ui.icon("folder", color="amber")
                        with ui.item_section():
                            ui.item_label(entry.name)

                elif entry.is_file():
                    if self.allowed_types is not None and entry.suffix.lower() not in self.allowed_types:
                        continue
                    with ui.item(on_click=lambda p=entry: self._pick_file(p)).classes(
                        "w-full"
                    ):
                        with ui.item_section().props("avatar"):
                            ui.icon("description", color="blue-grey")
                        with ui.item_section():
                            ui.item_label(entry.name)

        self.path_display.text = str(self.path)

    def _pick_file(self, path: Path) -> None:
        self.selected_file = path
        self.path_display.text = str(path)
        self.select_btn.enable()

    def _select(self) -> None:
        if self.selected_file is not None:
            self.submit(str(self.selected_file))
