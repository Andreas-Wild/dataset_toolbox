import numpy as np
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
    distances = np.sum((non_zeros - [x, y]) ** 2, axis=1)
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


if __name__ == "__main__":
    print(canvas)
    print("=" * 40)
    growed = grow_mask(canvas, mask_id)
    print(growed)
