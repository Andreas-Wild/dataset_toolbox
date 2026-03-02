"""Label Studio page — embeds Label Studio UI in an iframe."""

import os

from nicegui import ui

from src.layout import page_layout

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "/ls/")


@ui.page("/labelstudio")
def label_studio_page():
    with page_layout("Label Studio"):
        ui.html(
            f'<iframe src="{LABEL_STUDIO_URL}" '
            f'style="width:100%; height:calc(100vh - 64px); border:none;">'
            f"</iframe>",
            sanitize=False,
        ).classes("w-full")
