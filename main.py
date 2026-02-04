import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

from data_saver import DatasetSaver
from tools import grow_mask, pixel_colour, merge_mask

# Page config
st.set_page_config(
    page_title="Dataset Mask Editor",
    page_icon="🎨",
    layout="wide",
)


def load_dataset(dataset_path: str, saver: DatasetSaver | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load all images and masks from dataset, skipping already processed ones."""
    base_dir = Path(dataset_path)
    image_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    samples = []
    for file in sorted(image_dir.glob("*.png")):
        filename = file.name

        # Skip if already processed
        if saver and saver.is_processed(filename):
            continue

        try:
            img = np.asarray(Image.open(image_dir / filename))
            mask = np.asarray(Image.open(mask_dir / filename))
            samples.append((filename, img, mask))
        except FileNotFoundError:
            st.warning(f"Missing mask for: {filename}")

    return samples


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay of mask on image with colored regions."""
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = image.copy()

    # Create a colored overlay only where mask is non-zero
    mask_colour = cv2.applyColorMap((mask * 40).astype(np.uint8), cv2.COLORMAP_JET)
    mask_colour = cv2.cvtColor(mask_colour, cv2.COLOR_BGR2RGB)

    # Only apply color where mask has values (not background)
    mask_binary = (mask > 0).astype(np.uint8)
    mask_binary_3ch = np.stack([mask_binary] * 3, axis=-1) * 255

    overlayed_image = np.where(
        mask_binary_3ch, cv2.addWeighted(img_rgb, 0.5, mask_colour, 0.5, 0), img_rgb
    )
    return overlayed_image.astype(np.uint8)


def get_mask_id_at_position(mask: np.ndarray | None, x: int, y: int) -> int | None:
    """Get the mask ID at the given position."""
    if mask is None:
        return None
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        mask_id = int(mask[y, x])
        return mask_id if mask_id > 0 else None
    return None


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "dataset_path": "",
        "output_path": "",
        "image_list": [],
        "current_index": 0,
        "current_mask": None,
        "mask_history": [],
        "last_click": None,
        "saver": None,
        "dataset_loaded": False,
        "log_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_action(message: str, level: str = "info"):
    """Add a message to the log."""
    st.session_state.log_messages.append({"message": message, "level": level})
    # Keep only last 10 messages
    st.session_state.log_messages = st.session_state.log_messages[-10:]


def get_current_sample() -> tuple[str, np.ndarray, np.ndarray] | None:
    """Get the current image sample."""
    if not st.session_state.image_list:
        return None
    idx = st.session_state.current_index
    if 0 <= idx < len(st.session_state.image_list):
        return st.session_state.image_list[idx]
    return None


def navigate_to_image(index: int):
    """Navigate to a specific image index."""
    if 0 <= index < len(st.session_state.image_list):
        st.session_state.current_index = index
        _, _, mask = st.session_state.image_list[index]
        st.session_state.current_mask = mask.copy()
        st.session_state.mask_history = [mask.copy()]
        st.session_state.last_click = None


# Initialize session state
init_session_state()

# ============ SIDEBAR ============
with st.sidebar:
    st.title("🎨 Dataset Mask Editor")
    st.divider()

    # Path configuration
    st.subheader("📁 Paths")
    dataset_path = st.text_input(
        "Dataset Path",
        value=st.session_state.dataset_path,
        placeholder="/path/to/dataset/",
    )
    output_path = st.text_input(
        "Output Path",
        value=st.session_state.output_path,
        placeholder="/path/to/output/",
    )

    if st.button("📂 Load Dataset", use_container_width=True, type="primary"):
        if dataset_path and output_path:
            st.session_state.dataset_path = dataset_path
            st.session_state.output_path = output_path
            st.session_state.saver = DatasetSaver(output_path)
            st.session_state.image_list = load_dataset(dataset_path, st.session_state.saver)
            st.session_state.current_index = 0
            st.session_state.dataset_loaded = True

            if st.session_state.image_list:
                _, _, mask = st.session_state.image_list[0]
                st.session_state.current_mask = mask.copy()
                st.session_state.mask_history = [mask.copy()]
                log_action(f"Loaded {len(st.session_state.image_list)} images")
            else:
                log_action("No images found or all already processed", "warning")
            st.rerun()
        else:
            st.error("Please provide both paths")

    st.divider()

    # Navigation
    if st.session_state.dataset_loaded and st.session_state.image_list:
        st.subheader("🧭 Navigation")
        total = len(st.session_state.image_list)
        current = st.session_state.current_index + 1

        st.progress(current / total, text=f"Image {current} of {total}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀ Previous", use_container_width=True, disabled=st.session_state.current_index <= 0):
                navigate_to_image(st.session_state.current_index - 1)
                st.rerun()
        with col2:
            if st.button("Next ▶", use_container_width=True, disabled=st.session_state.current_index >= total - 1):
                navigate_to_image(st.session_state.current_index + 1)
                st.rerun()

        st.divider()

        # Actions
        st.subheader("⚡ Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                sample = get_current_sample()
                if sample and st.session_state.saver:
                    name, img, _ = sample
                    new_filename = st.session_state.saver.save_sample(
                        img, st.session_state.current_mask, original_name=name
                    )
                    log_action(f"✓ Saved as: {new_filename}", "success")

                    # Move to next image
                    if st.session_state.current_index < total - 1:
                        navigate_to_image(st.session_state.current_index + 1)
                    st.rerun()

        with col2:
            if st.button("⏭ Skip", use_container_width=True):
                sample = get_current_sample()
                if sample:
                    log_action(f"⊘ Skipped: {sample[0]}", "warning")
                    if st.session_state.current_index < total - 1:
                        navigate_to_image(st.session_state.current_index + 1)
                    st.rerun()

        st.divider()

        # Mask editing tools
        st.subheader("🛠️ Mask Tools")

        # Click position input
        st.caption("Click Position (for Grow Mask & Pixel Color)")
        click_col1, click_col2 = st.columns(2)
        with click_col1:
            click_x = st.number_input("X", min_value=0, value=st.session_state.last_click[0] if st.session_state.last_click else 0, key="click_x")
        with click_col2:
            click_y = st.number_input("Y", min_value=0, value=st.session_state.last_click[1] if st.session_state.last_click else 0, key="click_y")
        
        if st.button("📍 Set Click Position", use_container_width=True):
            st.session_state.last_click = (int(click_x), int(click_y))
            mask_id = get_mask_id_at_position(st.session_state.current_mask, int(click_x), int(click_y))
            if mask_id:
                log_action(f"📍 Position set: ({click_x}, {click_y}) - Mask ID: {mask_id}")
            else:
                log_action(f"📍 Position set: ({click_x}, {click_y}) - Background")
            st.rerun()

        if st.button("🎨 Color Pixel", use_container_width=True):
            if st.session_state.last_click and st.session_state.current_mask is not None:
                x, y = st.session_state.last_click
                mask_id = get_mask_id_at_position(st.session_state.current_mask, x, y)
                if mask_id is None:
                    st.session_state.mask_history.append(st.session_state.current_mask.copy())
                    st.session_state.current_mask = pixel_colour(st.session_state.current_mask.copy(), y, x)
                    log_action(f"🎨 Colored pixel at ({x}, {y})")
                    st.rerun()
                else:
                    log_action("Pixel already has a mask color", "warning")
            else:
                log_action("Set a click position first", "warning")

        if st.button("🌱 Grow Mask", use_container_width=True):
            if st.session_state.last_click:
                x, y = st.session_state.last_click
                mask_id = get_mask_id_at_position(st.session_state.current_mask, x, y)
                if mask_id is not None:
                    st.session_state.mask_history.append(st.session_state.current_mask.copy())
                    st.session_state.current_mask = grow_mask(st.session_state.current_mask, mask_id)
                    log_action(f"🌱 Grew mask ID: {mask_id}")
                    st.rerun()
                else:
                    log_action("Cannot grow: clicked on background", "warning")
            else:
                log_action("Set a click position first", "warning")

        if st.button("🔗 Merge Masks", use_container_width=True):
            st.session_state.mask_history.append(st.session_state.current_mask.copy())
            st.session_state.current_mask = merge_mask(st.session_state.current_mask)
            log_action("🔗 Merged closest masks")
            st.rerun()

        if st.button("↩ Undo", use_container_width=True):
            if len(st.session_state.mask_history) > 1:
                st.session_state.mask_history.pop()
                st.session_state.current_mask = st.session_state.mask_history[-1].copy()
                log_action(f"↩ Undo (history: {len(st.session_state.mask_history)})")
                st.rerun()
            else:
                log_action("Cannot undo: at original state", "warning")

        st.divider()

        # Log display
        st.subheader("📋 Activity Log")
        for msg in reversed(st.session_state.log_messages):
            if msg["level"] == "success":
                st.success(msg["message"])
            elif msg["level"] == "warning":
                st.warning(msg["message"])
            else:
                st.info(msg["message"])

    # Help section
    with st.expander("❓ Help"):
        st.markdown("""
        **Workflow:**
        1. Set dataset and output paths
        2. Click "Load Dataset"
        3. Enter X/Y coordinates to select pixels
        4. Use tools to modify masks
        5. Save or Skip to continue

        **Tools:**
        - **Color Pixel**: Colors background pixel with nearest mask color
        - **Grow Mask**: Expands the selected mask by 1 pixel
        - **Merge Masks**: Combines the two closest masks
        - **Undo**: Reverts the last change

        **Tips:**
        - Enter coordinates and click "Set Click Position"
        - Use the image display to find coordinates (hover shows position in most browsers)
        """)


# ============ MAIN AREA ============
st.title("Dataset Mask Editor")

if not st.session_state.dataset_loaded:
    st.info("👈 Configure paths and load dataset from the sidebar to begin")
elif not st.session_state.image_list:
    st.warning("No images to process. All images may have been processed already.")
else:
    sample = get_current_sample()
    if sample:
        name, img, original_mask = sample
        st.subheader(f"📷 {name}")

        # Create overlay image
        current_mask = st.session_state.current_mask
        if current_mask is None:
            current_mask = original_mask.copy()
            st.session_state.current_mask = current_mask
            st.session_state.mask_history = [current_mask.copy()]

        overlay_img = overlay(img, current_mask)

        # Display options
        col1, col2 = st.columns([3, 1])
        with col2:
            show_overlay = st.checkbox("Show Mask Overlay", value=True)
            show_original = st.checkbox("Show Original", value=False)

        # Main image display
        if show_original and show_overlay:
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.caption("Original")
                st.image(img, use_container_width=True)
            with img_col2:
                st.caption("With Mask Overlay")
                st.image(overlay_img, use_container_width=True)
        elif show_overlay:
            st.image(overlay_img, use_container_width=True)
        else:
            st.image(img, use_container_width=True)

        # Display current click info
        if st.session_state.last_click:
            x, y = st.session_state.last_click
            mask_id = get_mask_id_at_position(current_mask, x, y)
            st.caption(f"📍 Selected position: ({x}, {y}) | Mask ID: {mask_id if mask_id else 'background'}")

        # Show mask statistics
        with st.expander("📊 Mask Statistics"):
            unique_ids = np.unique(current_mask)
            unique_ids = unique_ids[unique_ids > 0]
            st.write(f"**Unique mask IDs:** {len(unique_ids)}")
            if len(unique_ids) > 0:
                st.write(f"**IDs:** {', '.join(map(str, unique_ids))}")
                for mask_id in unique_ids:
                    pixel_count = np.sum(current_mask == mask_id)
                    st.write(f"  - Mask {mask_id}: {pixel_count} pixels")
