"""Dataset Toolbox — centralised NiceGUI application.

Import all page modules (each registers its own route via @ui.page), then
start the NiceGUI server.
"""

import os
import secrets

from nicegui import ui

import src.pages.converter  # noqa: F401  –  /converter
import src.pages.editor  # noqa: F401  –  /editor

# Importing the page modules registers @ui.page routes as a side effect.
import src.pages.home  # noqa: F401  –  /
import src.pages.label_studio  # noqa: F401  –  /labelstudio
import src.pages.rle_converter  # noqa: F401  –  /rleconverter
import src.pages.viewer  # noqa: F401  –  /viewer

ui.run(
    title="Dataset Toolbox",
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8080")),
    reload=os.getenv("RELOAD", "false").lower() == "true",
    storage_secret=os.getenv("STORAGE_SECRET", secrets.token_urlsafe(16)),
    show=False,
)
