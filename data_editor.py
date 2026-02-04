from typing import Generator
from rich import print
from data_saver import DatasetSaver
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import os
from tools import grow_mask, pixel_colour, merge_mask

# Suppress Qt wanring
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

# Key bindings
KEY_SAVE = 9
KEY_SKIP = ord("n")  # Skip current image without saving
KEY_ESC = 27
KEY_GROW_MASK = ord("g")
KEY_MERGE_MASKS = ord("m")
KEY_UNDO = ord("u")

# Global variables
last_click_coords = None
current_mask = None
mask_history = []


def image_gen(
    dataset_path: str,
    saver: DatasetSaver | None = None,
) -> Generator[tuple[str, np.ndarray, np.ndarray], None, None]:
    """This generator iterates through a dataset and sequentially returns the filename, image and mask as a tuple."""
    base_dir = Path(dataset_path)
    image_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    skipped_count = 0
    total_count = 0

    for file in image_dir.glob("*.png"):
        filename = file.name
        total_count += 1

        # Skip if already processed
        if saver and saver.is_processed(filename):
            skipped_count += 1
            print(
                f"[dim]Skipping already processed: {filename} ({skipped_count} skipped so far)[/dim]"
            )
            continue

        try:
            img = Image.open(image_dir / filename)
            mask = Image.open(mask_dir / filename)
            yield (filename, np.asarray(img), np.asarray(mask))
        except FileNotFoundError as e:
            print(e)

    if skipped_count > 0:
        print(
            f"\n[cyan]Summary: Skipped {skipped_count}/{total_count} already processed images[/cyan]"
        )


def mouse_click_event(event, x, y, flags, param):
    """Callback function for mouse click events"""
    global last_click_coords, current_mask, mask_history
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click_coords = (x, y)
        img, window_name = param
        mask_id = get_mask_id_at_position(current_mask, x, y)
        if current_mask is not None and mask_id is None:
            mask_history.append(current_mask.copy())
            current_mask = pixel_colour(current_mask, y, x)
            update_window(window_name, img, current_mask)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image.copy()

    # Create a colored overlay only where mask is non-zero
    mask_colour = cv2.applyColorMap(mask * 40, cv2.COLORMAP_JET)

    # Only apply color where mask has values (not background)
    mask_binary = (mask > [0]).astype(np.uint8)
    mask_binary_3ch = cv2.cvtColor(mask_binary * 255, cv2.COLOR_GRAY2BGR)

    overlayed_image = np.where(
        mask_binary_3ch, cv2.addWeighted(img_bgr, 0.5, mask_colour, 0.5, 0), img_bgr
    )
    return overlayed_image


def get_help():
    print("\n[bold cyan]Key Bindings:[/bold cyan]")
    print("  [TAB] - Save and continue to next image")
    print("  [N] - Skip to next image without saving")
    print("  [R] - Refresh preview")
    print("  [G] - Grow mask by 1 pixel (at clicked position)")
    print("  [C] - Color pixel with nearest adjacent color (manual trigger)")
    print("  [M] - Merge contained masks")
    print("  [U] - Undo last change")
    print("  [H] - Show this help")
    print("  [ESC] - Exit")
    print(
        "\n[bold yellow]Note:[/bold yellow] Clicking on background (mask=0) automatically colors the pixel"
    )
    print(
        "\n[bold green]Auto-skip:[/bold green] Already processed images are automatically skipped"
    )


def update_window(window_name: str, image: np.ndarray, mask: np.ndarray):
    """Update the preview window with the modified mask overlay."""
    overlayed = overlay(image, mask)
    cv2.imshow(window_name, overlayed)


def get_mask_id_at_position(mask: np.ndarray | None, x: int, y: int) -> int | None:
    """
    Get the mask ID at the given position.

    :param mask: The mask array
    :type mask: np.ndarray
    :param x: X coordinate
    :type x: int
    :param y: Y coordinate
    :type y: int
    :return: The mask ID at the position, or None if out of bounds or background
    :rtype: int | None
    """
    if mask is None:
        return None

    elif 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        mask_id = int(mask[y, x])
        return mask_id if mask_id > 0 else None
    return None


def main():
    global last_click_coords, current_mask, mask_history
    dataset_path = (
        "/mnt/c/Users/andre/Desktop/masters_data/complete/dataset_2026_01_16/"
    )
    output_path = "/mnt/c/Users/andre/Desktop/masters_data/complete/dataset_edited/"
    saver = DatasetSaver(output_path)

    # Pass saver to image_gen to enable automatic skipping
    image_generator = image_gen(dataset_path, saver)

    # Create window
    window_name = "Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)

    for name, img, mask in image_generator:
        print(f"\n[bold]Processing: {name}[/bold]")
        current_mask = mask.copy()
        mask_history = [current_mask.copy()]
        overlayed_image = overlay(img, mask)
        cv2.imshow(window_name, overlayed_image)
        # Use the preview for clicking
        cv2.setMouseCallback(
            window_name,
            mouse_click_event,
            param=(img, window_name),
        )

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_SAVE:
                new_filename = saver.save_sample(img, current_mask, original_name=name)
                print(f"[green]✓ Saved as: {new_filename}[/green]")
                break

            elif key == KEY_SKIP:
                print(f"[yellow]⊘ Skipped without saving: {name}[/yellow]")
                break

            elif key == ord("h"):  # Get help
                get_help()

            elif key == KEY_MERGE_MASKS:
                mask_history.append(current_mask.copy())
                current_mask = merge_mask(current_mask)
                update_window(window_name, img, current_mask)

            elif key == KEY_GROW_MASK:
                mask_id = None
                if last_click_coords:
                    mask_id = get_mask_id_at_position(current_mask, *last_click_coords)
                    if mask_id is not None:
                        print(f"[cyan]Growing mask ID: {mask_id}[/cyan]")
                        mask_history.append(current_mask.copy())
                        current_mask = grow_mask(current_mask, mask_id)
                        update_window(window_name, img, current_mask)
                    else:
                        print(
                            "[yellow]Cannot grow: clicked on background (mask=0)[/yellow]"
                        )
                else:
                    print("[yellow]No position clicked yet![/yellow]")

            elif key == KEY_UNDO:
                if len(mask_history) > 1:
                    current_mask = mask_history.pop()
                    print(
                        f"[green]Undo successful. History size: {len(mask_history)}[/green]"
                    )
                else:
                    print("[yellow]Cannot undo: at original state[/yellow]")
                update_window(window_name, img, current_mask)

            elif key == 27:  # ESC key
                print("[red]Exiting...[/red]")
                cv2.destroyWindow(name)
                return

    cv2.destroyAllWindows()
    print("\n[bold green]✓ All images processed![/bold green]")


if __name__ == "__main__":
    main()
