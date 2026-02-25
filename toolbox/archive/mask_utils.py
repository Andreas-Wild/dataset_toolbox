import numpy as np
from PIL import Image
import io
import base64
from rich import print
import cv2

k = 10
canvas = np.zeros((k, k), dtype=np.uint8)
start, end, mask_id = (3, 8, 5)
if end < start:
    start, end = end, start
canvas[start:end, start:end] = (
    np.ones(canvas[start:end, start:end].shape, dtype=np.uint8) * mask_id
)


def grow_mask(mask: np.ndarray, mask_id: int | None = None) -> np.ndarray:
    """
    This function grows the input mask by one pixel. No diagonals.

    :param mask: The mask that should be grown
    :type mask: np.ndarray
    :param mask_id: The class or number of the mask int between (1...5)
    :type mask_id: int | None
    :return: Returns the updated mask after growing
    :return_type: ndarray[_AnyShape, dtype[Any]]
    """
    if mask_id is not None:
        # Create mask for id
        binary_mask = np.array((mask == mask_id), dtype=np.uint8)
        # Get SE and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)

        result = mask.copy()

        # Only update pixels that are currently 0 and are in the dilated region
        new_pixels = (dilated > 0) & (mask == 0)
        result[new_pixels] = mask_id

        return result
    else:
        return mask

def shrink_mask(mask: np.ndarray, mask_id: int | None = None) -> np.ndarray:
    """
    This function shrinks the input mask by one pixel. No diagonals.

    :param mask: The mask that should be shrunk
    :type mask: np.ndarray
    :param mask_id: The class or number of the mask int between (1...5)
    :type mask_id: int | None
    :return: Returns the updated mask after growing
    :return_type: ndarray[_AnyShape, dtype[Any]]
    """
    if mask_id is not None:
        # Create mask for id
        binary_mask = np.array((mask == mask_id), dtype=np.uint8)
        # Get SE and erode
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(binary_mask, kernel, iterations=1)

        result = mask.copy()

        # Update the mask_id pixels to match the eroded version
        result[mask == mask_id] = eroded[mask == mask_id] * mask_id

        return result
    else:
        return mask


def pixel_colour(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    non_zeros = np.argwhere(mask)

    # There shouldn't be empty images but just in case
    if len(non_zeros) == 0:
        return mask

    # Use euclidean distance
    distances = np.sum((non_zeros - [y, x]) ** 2, axis=1)
    nearest = np.argmin(distances)

    nearest_pixel_value = mask[tuple(non_zeros[nearest])]
    # Update the mask with this pixel value
    mask[y, x] = nearest_pixel_value
    return mask

def remove_pixel(mask: np.ndarray, x:int, y:int) -> np.ndarray:
    mask[y, x] = 0
    return mask


def merge_mask(mask: np.ndarray) -> np.ndarray:
    # Get all unique mask IDs (excluding background which is 0)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) < 2:
        print("[yellow]Need at least 2 masks to merge[/yellow]")
        return mask

    # Calculate minimum distance between all pairs of masks
    min_distance = float("inf")
    closest_pair = None

    for i, id1 in enumerate(unique_ids):
        for id2 in unique_ids[i + 1 :]:
            # Get all pixels for each mask
            pixels1 = np.argwhere(mask == id1)
            pixels2 = np.argwhere(mask == id2)

            # Calculate pairwise distances between all pixels of the two masks
            # Using broadcasting to compute all distances at once
            distances = np.sqrt(
                np.sum(
                    (pixels1[:, np.newaxis, :] - pixels2[np.newaxis, :, :]) ** 2, axis=2
                )
            )

            # Get the minimum distance between these two masks
            current_min = np.min(distances)

            if current_min < min_distance:
                min_distance = current_min
                closest_pair = (id1, id2)

    if closest_pair is not None:
        id1, id2 = closest_pair
        result = mask.copy()
        # Merge id2 into id1 (replace all id2 pixels with id1)
        result[mask == id2] = id1
        print(
            f"[green]Merged mask {id2} into mask {id1} (distance: {min_distance:.2f})[/green]"
        )
        return result

    return mask


def split_connected(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Split connected pixels at position (x, y) into a new mask ID.
    
    :param mask: The mask containing the pixels to split
    :type mask: np.ndarray
    :param x: The x coordinate of the clicked position
    :type x: int
    :param y: The y coordinate of the clicked position
    :type y: int
    :return: Returns the updated mask with split connected component
    :return_type: np.ndarray
    """
    # Check bounds
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return mask
    
    mask_id = mask[y, x]
    
    # If background, nothing to split
    if mask_id == 0:
        return mask
    
    # Create binary mask for the selected mask_id
    binary_mask = (mask == mask_id).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Find which component contains the clicked point
    clicked_label = labels[y, x]
    
    # If no connected region found (shouldn't happen but safety check)
    if clicked_label == 0:
        return mask
    
    # Find max mask ID in current mask
    max_id = int(mask.max())
    new_id = max_id + 1
    
    # Change only the connected component containing the clicked pixel to new ID
    result = mask.copy()
    connected_region = (labels == clicked_label)
    result[connected_region] = new_id
    
    print(f"[green]Split connected pixels to mask ID {new_id}[/green]")
    return result

def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay of mask on image with colored regions."""
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = image.copy()

    mask_colour = cv2.applyColorMap((mask * 40).astype(np.uint8), cv2.COLORMAP_JET)
    mask_colour = cv2.cvtColor(mask_colour, cv2.COLOR_BGR2RGB)

    mask_binary = (mask > 0).astype(np.uint8)
    mask_binary_3ch = np.stack([mask_binary] * 3, axis=-1) * 255

    overlayed = np.where(
        mask_binary_3ch,
        cv2.addWeighted(img_rgb, 0.5, mask_colour, 0.5, 0),
        img_rgb,
    )
    return overlayed.astype(np.uint8)

def array_to_base64(arr: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    print(canvas)
    print("=" * 40)
    growed = grow_mask(canvas, mask_id)
    print(growed)
