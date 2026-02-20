"""Reusable local directory picker dialog for NiceGUI.

Lets users browse the server's filesystem and select a directory,
instead of typing a path string manually.
"""

from pathlib import Path

from nicegui import ui


class LocalDirectoryPicker(ui.dialog):
    """A dialog that lets users navigate and pick a local directory."""

    def __init__(
        self,
        directory: str = "~",
        *,
        title: str = "Select Directory",
        show_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.path = Path(directory).expanduser().resolve()
        if not self.path.is_dir():
            self.path = Path.home()
        self.show_hidden = show_hidden

        with self, ui.card().classes("w-96"):
            ui.label(title).classes("text-lg font-semibold")
            self.path_display = ui.label(str(self.path)).classes(
                "text-l font-bold break-all"
            )
            self.dir_list = ui.scroll_area().classes("w-full border rounded")
            self.dir_list.style("height: 300px")

            with ui.row().classes("w-full justify-between gap-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                ui.button("Select", on_click=self._select).props("color=primary")

        self._update_list()
        self.open()

    def _update_list(self) -> None:
        self.dir_list.clear()
        with self.dir_list:
            # Parent directory entry
            parent = self.path.parent
            with ui.item(on_click=lambda p=parent: self._navigate(p)).classes("w-full"):
                with ui.item_section().props("avatar"):
                    ui.icon("folder", color="amber")
                with ui.item_section():
                    ui.item_label(text="⟵ Go Back")

            ui.separator()

            # Child directories
            try:
                entries = sorted(self.path.iterdir(), key=lambda p: p.name.lower())
            except PermissionError:
                entries = []
            for entry in entries:
                if entry.is_dir():
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

    def _navigate(self, path: Path) -> None:
        if path.is_dir():
            self.path = path
            self._update_list()

    def _select(self) -> None:
        self.submit(str(self.path))
