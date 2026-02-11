import io
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Literal

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Core load
# ─────────────────────────────────────────────────────────────────────────────
def load_pdf_page_rgb(pdf_path: str, page_num: int, zoom: float = 1.0):
    """
    Returns:
        img_rgb: HxWx3 uint8 (RGB)
        page_w_pts, page_h_pts: page size in POINTS (72 pt per inch)
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_rgb = np.array(img)  # RGB
    return img_rgb, float(page.rect.width), float(page.rect.height)



def _sanitize_filename(name: str, fallback: str = "symbol") -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or fallback

def _unique_path(base_dir: Path, stem: str, ext: str = ".png") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    p = base_dir / f"{stem}{ext}"
    if not p.exists():
        return p
    i = 1
    while True:
        p_try = base_dir / f"{stem}_{i:02d}{ext}"
        if not p_try.exists():
            return p_try
        i += 1

def prompt_symbol_name(default_stem: str) -> str:
    try:
        user = input(f"Enter symbol name (no extension) [{default_stem}]: ").strip()
    except Exception:
        user = ""
    stem = _sanitize_filename(user or default_stem)
    return stem or "symbol"

# ─────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ─────────────────────────────────────────────────────────────────────────────
PT_PER_IN = 72.0
MM_PER_IN = 25.4

def pts_to_px(v: float, zoom: float) -> float:
    return v * zoom

def px_to_pts(v: float, zoom: float) -> float:
    return v / zoom

def pts_to_mm(v: float) -> float:
    return v * (MM_PER_IN / PT_PER_IN)

def mm_to_pts(v: float) -> float:
    return v * (PT_PER_IN / MM_PER_IN)

def clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(int(round(x0)), w))
    x1 = max(0, min(int(round(x1)), w))
    y0 = max(0, min(int(round(y0)), h))
    y1 = max(0, min(int(round(y1)), h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return x0, y0, x1, y1

# ─────────────────────────────────────────────────────────────────────────────
# Friendly crop APIs
# ─────────────────────────────────────────────────────────────────────────────
def crop_by_corners(
    img_rgb: np.ndarray,
    zoom: float,
    box: Tuple[float, float, float, float],
    units: Literal["pts", "px", "mm"] = "pts",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if units == "mm":
        x0, y0, x1, y1 = [mm_to_pts(v) for v in box]
        x0, y0, x1, y1 = [pts_to_px(v, zoom) for v in (x0, y0, x1, y1)]
    elif units == "pts":
        x0, y0, x1, y1 = [pts_to_px(v, zoom) for v in box]
    elif units == "px":
        x0, y0, x1, y1 = box
    else:
        raise ValueError("units must be 'pts', 'px', or 'mm'")

    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
    crop = img_rgb[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)

def crop_by_center_size(
    img_rgb: np.ndarray,
    zoom: float,
    center: Tuple[float, float],
    size: Tuple[float, float],
    units: Literal["pts", "px", "mm"] = "pts",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cx, cy = center
    ww, hh = size

    if units == "mm":
        cx, cy = mm_to_pts(cx), mm_to_pts(cy)
        ww, hh = mm_to_pts(ww), mm_to_pts(hh)
    elif units == "px":
        cx, cy = px_to_pts(cx, zoom), px_to_pts(cy, zoom)
        ww, hh = px_to_pts(ww, zoom), px_to_pts(hh, zoom)

    x0_pts = cx - ww / 2
    y0_pts = cy - hh / 2
    x1_pts = cx + ww / 2
    y1_pts = cy + hh / 2

    x0, y0, x1, y1 = [pts_to_px(v, zoom) for v in (x0_pts, y0_pts, x1_pts, y1_pts)]
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
    crop = img_rgb[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)

# ─────────────────────────────────────────────────────────────────────────────
# Interactive cropping (mouse drag)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DragResult:
    box_px: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1) in pixels

def interactive_crop(img_rgb: np.ndarray, title: str = "Drag to crop") -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    result = DragResult()

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img_rgb)
    ax.set_title(title)
    ax.axis("on")

    def on_select(eclick, erelease):
        x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
        H, W = img_rgb.shape[:2]
        result.box_px = clamp_box(x0, y0, x1, y1, W, H)

    rs = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=3, minspany=3,
        spancoords="pixels", interactive=True
    )

    plt.show()

    if result.box_px is None:
        raise RuntimeError("No crop selected.")
    x0, y0, x1, y1 = result.box_px
    crop = img_rgb[y0:y1, x0:x1].copy()
    return crop, result.box_px

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Interactive masking on the crop
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MaskSession:
    rects: List[Tuple[int, int, int, int]]
    finished: bool = False
    canceled: bool = False

def interactive_mask_on_crop(
    crop_rgb: np.ndarray,
    title: str = "Mask text: drag rectangles (Enter=finish, u=undo, c=clear, p=preview, Esc=cancel)"
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Lets user draw multiple rectangles over the crop. Returns (masked_image, rects_px).
    - Left-drag adds a rectangle (immediately previewed with a translucent patch).
    - Press 'u' to undo last rectangle.
    - Press 'c' to clear all rectangles.
    - Press 'p' to preview current masked result in a separate window.
    - Press 'Enter' to finish and apply masks.
    - Press 'Esc' to cancel (returns original crop and empty list).
    """
    sess = MaskSession(rects=[])

    # Working copy to show overlays (but not destructive)
    overlay_img = crop_rgb.copy()
    H, W = crop_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(overlay_img)
    ax.set_title(title)
    ax.axis("off")

    # Store patches so we can remove on undo/clear
    patches = []

    def add_rect_patch(x0, y0, x1, y1):
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=True, alpha=0.25, edgecolor="k", linewidth=0.8, facecolor="white"
        )
        ax.add_patch(rect)
        patches.append(rect)

    def redraw_overlay():
        im.set_data(overlay_img)
        fig.canvas.draw_idle()
    
    def _inflate_and_clamp_rect(x0, y0, x1, y1, W, H, inflate=2):
        x0 = max(0, x0 - inflate)
        y0 = max(0, y0 - inflate)
        x1 = min(W, x1 + inflate)
        y1 = min(H, y1 + inflate)
        return x0, y0, x1, y1

    def _snap_to_edges(x0, y0, x1, y1, W, H, eps=4):
        if x0 <= eps: x0 = 0
        if y0 <= eps: y0 = 0
        if W - x1 <= eps: x1 = W
        if H - y1 <= eps: y1 = H
        return x0, y0, x1, y1


    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        sess.rects.append((x0, y0, x1, y1))
        add_rect_patch(x0, y0, x1, y1)
        # live preview (non-destructive): draw a white rectangle on a temp copy
        temp = overlay_img.copy()
        temp[y0:y1, x0:x1] = (255, 255, 255)
        im.set_data(temp)
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        if key == "enter":
            sess.finished = True
            plt.close(fig)
        elif key == "escape":
            sess.canceled = True
            plt.close(fig)
        elif key == "u":
            if sess.rects:
                sess.rects.pop()
            if patches:
                p = patches.pop()
                p.remove()
            # recompute live overlay from scratch
            temp = crop_rgb.copy()
            for (x0, y0, x1, y1) in sess.rects:
                temp[y0:y1, x0:x1] = (255, 255, 255)
            im.set_data(temp)
            fig.canvas.draw_idle()
        elif key == "c":
            sess.rects.clear()
            # remove all patches
            while patches:
                patches.pop().remove()
            im.set_data(crop_rgb)
            fig.canvas.draw_idle()
        elif key == "p":
            # quick preview in a separate figure (non-blocking)
            prev = crop_rgb.copy()
            for (x0, y0, x1, y1) in sess.rects:
                prev[y0:y1, x0:x1] = (255, 255, 255)
            plt.figure("Preview (current masks)")
            plt.imshow(prev)
            plt.axis("off")
            plt.show(block=False)

    rs = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=3, minspany=3,
        spancoords="pixels", interactive=True
    )
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if sess.canceled:
        return crop_rgb.copy(), []
    # Apply masks to a copy
# (After plt.show() and after handling sess.canceled / sess.rects)

    H, W = crop_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    for (x0, y0, x1, y1) in sess.rects:
        x0, y0, x1, y1 = _snap_to_edges(x0, y0, x1, y1, W, H, eps=4)
        x0, y0, x1, y1 = _inflate_and_clamp_rect(x0, y0, x1, y1, W, H, inflate=2)
        mask[y0:y1, x0:x1] = 255

    # Optional: quick preview window that uses the mask (no edge bleed)
    prev = crop_rgb.copy()
    prev[mask == 255] = (255, 255, 255)
    plt.figure("Preview (final mask)"); plt.imshow(prev); plt.axis("off"); plt.show(block=False)

    masked = crop_rgb.copy()
    masked[mask == 255] = (255, 255, 255)
    # Return original crop and the binary mask
    return masked, mask, sess.rects


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────
def resize_and_apply_mask(crop_rgb: np.ndarray, mask: np.ndarray, scale: float):
    new_w = max(1, int(round(crop_rgb.shape[1] * scale)))
    new_h = max(1, int(round(crop_rgb.shape[0] * scale)))
    img_resized  = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask,      (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = img_resized.copy()
    out[mask_resized == 255] = (255, 255, 255)
    return out

def mask_rectangles_inplace(img_rgb, rects_px: Iterable[Tuple[int, int, int, int]], color=(255, 255, 255)):
    for (x0, y0, x1, y1) in rects_px:
        img_rgb[y0:y1, x0:x1] = color

def resize_image(img_rgb, scale: Optional[float] = None, size: Optional[Tuple[int, int]] = None):
    if (scale is None) == (size is None):
        raise ValueError("Provide exactly one of 'scale' or 'size'.")
    if scale is not None:
        new_w = max(1, int(round(img_rgb.shape[1] * scale)))
        new_h = max(1, int(round(img_rgb.shape[0] * scale)))
        size = (new_w, new_h)
    return cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)

def show_centered_origin(img_rgb, title="Centered Origin", grid=True):
    h, w = img_rgb.shape[:2]
    extent = (-w / 2, w / 2, -h / 2, h / 2)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb, extent=extent, origin="upper")
    if grid:
        plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.axhline(0, color="red", lw=1)
    plt.axvline(0, color="red", lw=1)
    plt.title(title)
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PDF = "PDF\C2 LV Schematics.pdf"
    PAGE_NUM = 1- 1
    ZOOM = 3.0

    img_rgb, page_w_pts, page_h_pts = load_pdf_page_rgb(PDF, PAGE_NUM, ZOOM)

    # Select region interactively
    crop, box_px = interactive_crop(img_rgb, title=f"Page {PAGE_NUM} @ zoom={ZOOM} — drag to crop")

    # NEW: mask text/labels inside the crop by dragging rectangles
    masked, mask, rects = interactive_mask_on_crop(
        crop, title="Mask text: drag rectangles (Enter=finish, u=undo, c=clear, p=preview, Esc=cancel)"
    )

    # Optional: resize (undo zoom, etc.)
    crop_small = resize_and_apply_mask(crop, mask, scale=1.0 / ZOOM)

    # (Optional) visualize centered axes
    # show_centered_origin(crop_small, title="Cropped & masked (centered axes)", grid=True)

    # Save
    x0, y0, x1, y1 = box_px
    box_pts = tuple(px_to_pts(v, ZOOM) for v in (x0, y0, x1, y1))
    default_stem = f"sym_p{PAGE_NUM+1}_x{int(box_pts[0])}_y{int(box_pts[1])}"

    stem = prompt_symbol_name(default_stem)
    save_path = _unique_path(Path("symbols"), stem, ".png")
    cv2.imwrite(str(save_path), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {save_path}")

    # Report boxes in multiple units
    box_mm  = tuple(pts_to_mm(v) for v in box_pts)
    print("Crop box (px):", box_px)
    print("Crop box (pts):", tuple(round(v, 2) for v in box_pts))
    print("Crop box (mm): ", tuple(round(v, 2) for v in box_mm))
    print("Masks (px):", mask)
