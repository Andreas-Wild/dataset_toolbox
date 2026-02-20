"""A NiceGUI element that displays the contents of a directory as a table."""

from pathlib import Path

from nicegui import ui


def _row_from_path(p: Path) -> dict:
    return {
        "name": p.name,
        "type": "Directory" if p.is_dir() else p.suffix.lstrip(".").upper() or "File",
        "parent": str(p.parent),
    }


COLUMNS = [
    {
        "name": "name",
        "label": "Name",
        "field": "name",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "type",
        "label": "Type",
        "field": "type",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "parent",
        "label": "Parent Directory",
        "field": "parent",
        "sortable": True,
        "align": "left",
    },
]


class ListDir(ui.table):
    """Displays the contents of a directory as a NiceGUI table."""

    def __init__(
        self, directory: str | Path | None = None, rows_per_page: int = 20
    ) -> None:
        super().__init__(
            columns=COLUMNS, rows=[], row_key="name", pagination=rows_per_page
        )
        self.classes("w-full")
        self.props("flat bordered dense")
        if directory:
            self.load(directory)

    def load(self, directory: str | Path) -> None:
        """Populate the table from the given directory path."""
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            self.rows.clear()
            self.update()
            return

        rows = []
        try:
            for entry in sorted(
                path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            ):
                try:
                    rows.append(_row_from_path(entry))
                except PermissionError:
                    pass
        except PermissionError:
            pass

        self.rows.clear()
        self.rows.extend(rows)
        self.update()
