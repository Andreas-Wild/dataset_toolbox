"""Dataset Toolbox — centralised NiceGUI application.

Import all page modules (each registers its own route via @ui.page), then
start the NiceGUI server.
"""

import secrets

from nicegui import ui

import src.pages.converter  # noqa: F401  –  /converter
import src.pages.editor  # noqa: F401  –  /editor

# Importing the page modules registers @ui.page routes as a side effect.
import src.pages.home  # noqa: F401  –  /
import src.pages.viewer  # noqa: F401  –  /viewer

# Use for multithreaded process, like when native=true and reload=false
# if __name__ == "__main__":
#     ui.run(title="Dataset Toolbox", reload=True)

ui.run(title="Dataset Toolbox", reload=True, storage_secret=secrets.token_urlsafe(16))
