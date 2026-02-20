"""Dataset Toolbox — centralised NiceGUI application.

Import all page modules (each registers its own route via @ui.page), then
start the NiceGUI server.
"""

from nicegui import ui

# Importing the page modules registers @ui.page routes as a side effect.
import src.pages.home    # noqa: F401  –  /
import src.pages.editor  # noqa: F401  –  /editor
import src.pages.viewer  # noqa: F401  –  /viewer
import src.pages.converter  # noqa: F401  –  /converter

# Use for multithreaded process, like when native=true and reload=false
# if __name__ == "__main__":
#     ui.run(title="Dataset Toolbox", reload=True)

ui.run(title="Dataset Toolbox", reload=True)