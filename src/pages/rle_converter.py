"""RLE Converter page — convert Label Studio brush labels to png masks."""

import asyncio
import random
from pathlib import Path

import cv2
from nicegui import ui

from src.components.lightbox import Lightbox
from src.components.list_dir import ListDir
from src.components.local_picker import LocalDirectoryPicker, LocalFilePicker
from src.layout import page_layout
from toolbox.rle_converter import RLEConverter
from toolbox.utilities import array_to_base64, overlay

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PIXELATED_STYLE = "image-rendering: pixelated;"
GALLERY_SIZE = 9

PAGE_HELP = """
The RLE Converter may be used to convert Label Studio brush labels in the RLE format to `png` images. Select the annotations.json file and click convert.
"""


def _build_overlay_urls(masks_dir: Path, source_dir: Path) -> list[str]:
    """Pre-compute base64 data URLs for all mask/source overlay images."""
    urls: list[str] = []
    for mask_path in sorted(masks_dir.iterdir()):
        if mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        source_path = source_dir / mask_path.name
        if source_path.is_file():
            source_img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if source_img is not None:
                if len(source_img.shape) == 2:
                    source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)
                elif source_img.shape[2] == 4:
                    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGRA2RGB)
                else:
                    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                if mask.shape[:2] != source_img.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (source_img.shape[1], source_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                display = overlay(source_img, mask)
            else:
                display = _mask_to_rgb(mask)
        else:
            display = _mask_to_rgb(mask)

        b64 = array_to_base64(display)
        urls.append(f"data:image/png;base64,{b64}")
    return urls


def _show_gallery_page(
    urls: list[str], lightbox: Lightbox, grid: ui.grid
) -> None:
    """Display up to GALLERY_SIZE images in the grid."""
    lightbox.clear()
    grid.clear()
    sample = random.sample(urls, min(GALLERY_SIZE, len(urls)))
    with grid:
        for data_url in sample:
            lightbox.add_image(
                thumb_url=data_url,
                orig_url=data_url,
            ).style(PIXELATED_STYLE).classes("w-full h-[200px]")


def _mask_to_rgb(mask):
    """Convert a mask array to a visible RGB image as a fallback."""
    if mask.max() > 0:
        mask = (mask.astype(float) / mask.max() * 255).astype("uint8")
    if len(mask.shape) == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    elif mask.shape[2] == 4:
        return cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)


@ui.page("/rleconverter")
def rleconverter_page():
    with page_layout("RLE Converter", help_text=PAGE_HELP):
        main_container = ui.column(wrap=False).classes("w-full h-full")
        with main_container:
            with ui.column().classes("w-full justify-center items-center"):

                async def run_conversion():
                    annotations_path = annotations_label.text
                    image_source_path = image_source_label.text
                    if (
                        annotations_path == "No file selected"
                        or image_source_path == "No directory selected"
                    ):
                        ui.notify(
                            "Please select the annotations file and image source directory.",
                            type="warning",
                        )
                        return
                    convert_btn.disable()
                    progress_bar.visible = True
                    status_label.set_text("Converting...")
                    progress_bar.set_value(0)

                    try:
                        converter = RLEConverter(
                            annotation_input_json=annotations_path,
                            image_source_path=image_source_path,
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None, converter.prepare_dataset
                        )
                        status_label.set_text("Conversion complete!")
                        progress_bar.set_value(1)
                        ui.notify("Conversion complete!", type="positive")
                        # Show the output masks directory
                        output_dir = Path(annotations_path).parent / "masks"
                        output_section.set_visibility(True)
                        file_table.load(str(output_dir))
                        # Build overlay images and show a random sample
                        all_urls.clear()
                        all_urls.extend(
                            _build_overlay_urls(output_dir, Path(image_source_path))
                        )
                        _show_gallery_page(all_urls, lightbox, gallery_grid)
                        shuffle_btn.set_visibility(len(all_urls) > GALLERY_SIZE)
                    except Exception as e:
                        status_label.set_text(f"Error: {e}")
                        ui.notify(f"Conversion failed: {e}", type="negative")
                    finally:
                        convert_btn.enable()

            with ui.column().classes("w-full items-center"):
                with ui.splitter(value=40, limits=(0, 40)).classes(
                    "w-full h-full"
                ) as splitter:
                    # Sidebar
                    with splitter.before:

                        async def pick_annotations_file():
                            start_dir = (
                                str(Path(annotations_label.text).parent)
                                if annotations_label.text != "No file selected"
                                else "~"
                            )
                            result: str = await LocalFilePicker(
                                start_dir,
                                allowed_types=[".json"],
                                title="Select annotations.json",
                            )
                            if result:
                                annotations_label.set_text(result)

                                # Try to read image_source.txt next to the annotations file
                                source_txt = Path(result).parent / "image_source.txt"
                                if source_txt.is_file():
                                    source_path = source_txt.read_text().strip()
                                    if source_path and Path(source_path).is_dir():
                                        image_source_label.set_text(source_path)
                                        image_source_section.set_visibility(False)
                                        convert_section.set_visibility(True)
                                        ui.notify(
                                            "Image source loaded from image_source.txt",
                                            type="info",
                                        )
                                    else:
                                        ui.notify(
                                            "image_source.txt found but path is invalid. Please select manually.",
                                            type="warning",
                                        )
                                        image_source_section.set_visibility(True)
                                        await pick_image_source_dir()
                                else:
                                    ui.notify(
                                        "No image_source.txt found. Please select the image source directory.",
                                        type="warning",
                                    )
                                    image_source_section.set_visibility(True)
                                    await pick_image_source_dir()

                        async def pick_image_source_dir():
                            start_dir = (
                                image_source_label.text
                                if image_source_label.text != "No directory selected"
                                else "~"
                            )
                            result: str = await LocalDirectoryPicker(
                                start_dir,
                                title="Select Image Source Directory",
                            )
                            if result:
                                image_source_label.set_text(result)
                                convert_section.set_visibility(True)

                        with ui.column().classes("w-full items-center justify-center"):
                            # --- Input section ---
                            ui.label("Input").classes("text-lg font-semibold")
                            ui.separator()

                            with ui.row().classes("gap-2 justify-center items-center"):
                                ui.button(
                                    text="Annotations.json",
                                    icon="description",
                                    on_click=pick_annotations_file,
                                ).props("flat dense").tooltip("Select annotations.json")
                                annotations_label = ui.label(
                                    "No file selected"
                                ).classes("text-m text-gray-500")

                            image_source_section = ui.column().classes(
                                "w-full items-center"
                            )
                            image_source_section.set_visibility(False)
                            with image_source_section:
                                ui.separator()
                                ui.label("Image Source").classes(
                                    "text-lg font-semibold"
                                )
                                with ui.row().classes(
                                    "gap-2 justify-center items-center"
                                ):
                                    ui.button(
                                        text="Image source",
                                        icon="folder",
                                        on_click=pick_image_source_dir,
                                    ).props("flat dense").tooltip(
                                        "Select image source directory"
                                    )
                                    image_source_label = ui.label(
                                        "No directory selected"
                                    ).classes("text-m text-gray-500")

                            # --- Convert section ---
                            convert_section = ui.column().classes("w-full items-center")
                            convert_section.set_visibility(False)
                            with convert_section:
                                ui.separator()
                                ui.label("Convert").classes("text-lg font-semibold")
                                convert_btn = ui.button(
                                    icon="swap_horizontal_circle",
                                    text="Convert",
                                    on_click=run_conversion,
                                ).classes("text-xl icon-xl hover:animate-pulse")
                                progress_bar = ui.linear_progress(
                                    value=0, show_value=False
                                ).classes("w-3/4")
                                progress_bar.visible = False
                                status_label = ui.label("").classes(
                                    "text-sm text-gray-400"
                                )

                            # --- Output section ---
                            output_section = ui.column().classes("w-full items-center")
                            output_section.set_visibility(False)
                            with output_section:
                                ui.separator()
                                ui.label("Output").classes("text-lg font-semibold")
                                with ui.scroll_area().classes("w-full").style(
                                    "max-height: 400px"
                                ):
                                    file_table = ListDir()

                    with splitter.after:
                        all_urls: list[str] = []
                        lightbox = Lightbox()
                        shuffle_btn = ui.button(
                            "Shuffle",
                            icon="shuffle",
                            on_click=lambda: _show_gallery_page(
                                all_urls, lightbox, gallery_grid
                            ),
                        ).props("flat dense")
                        shuffle_btn.set_visibility(False)
                        with ui.scroll_area().classes("w-full").style(
                            "height: 80vh"
                        ):
                            gallery_grid = ui.grid(columns=3).classes(
                                "w-full gap-2 p-2"
                            )
