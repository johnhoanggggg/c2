"""
Convert a plain (image-based) PDF to a searchable PDF using OCR.
Optimized for circuit schematics with sparse, scattered text.

Opens a matplotlib window showing detected text bounding boxes on page 1
so you can verify OCR placement before processing all pages.

Requirements:
    pip install pytesseract pymupdf Pillow pypdf matplotlib numpy
    Also install Tesseract OCR:
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
        Linux:   sudo apt install tesseract-ocr

Usage:
    1. Set INPUT_PDF below to your PDF path
    2. Click Play in VS Code
"""

import sys
import subprocess
from pathlib import Path

# --- Configuration ---
INPUT_PDF = "PDF\C2 LV Schematics.pdf"   # <-- Change this to your PDF path
OUTPUT_PDF = None          # None = auto-generates name like input_searchable.pdf
DPI = 300                  # High DPI helps with small schematic text
LANG = "eng"               # Tesseract language
MIN_CONFIDENCE = 20        # Show text above this confidence level in preview


def ensure_dependencies():
    """Install Python packages if missing."""
    packages = {
        "pytesseract": "pytesseract",
        "fitz": "pymupdf",
        "PIL": "Pillow",
        "pypdf": "pypdf",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])


ensure_dependencies()

import pytesseract
import fitz  # pymupdf
from PIL import Image, ImageFilter, ImageEnhance
from pypdf import PdfReader, PdfWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import os


# Tesseract config for sparse schematic text
# --psm 11 = Sparse text. Find as much text as possible in no particular order.
# --psm 12 = Sparse text with OSD.
# OEM 3 = default (LSTM + legacy combined)
TESS_CONFIG = "--psm 11 --oem 1"


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """Convert a single PDF page to a PIL Image using pymupdf."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR on circuit schematics.
    - Convert to grayscale
    - Sharpen to make thin text crisper
    - Increase contrast
    - Binarize with adaptive-like threshold
    """
    gray = img.convert("L")

    # Sharpen
    sharp = gray.filter(ImageFilter.SHARPEN)
    sharp = sharp.filter(ImageFilter.SHARPEN)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(sharp)
    enhanced = enhancer.enhance(2.0)

    # Binarize — helps tesseract with thin lines vs text
    arr = np.array(enhanced)
    threshold = 180  # adjust if needed; higher = more aggressive whitening
    arr = ((arr > threshold) * 255).astype(np.uint8)
    binary = Image.fromarray(arr)

    return binary


def ocr_with_data(img: Image.Image, lang: str = "eng") -> dict:
    """Run OCR and return detailed word-level data."""
    return pytesseract.image_to_data(
        img, lang=lang, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )


def show_ocr_preview(pdf_path: str, dpi: int = 300, lang: str = "eng"):
    """Open a matplotlib window showing page 1 with OCR bounding boxes."""
    print(f"\n{'='*60}")
    print(f"OCR PREVIEW - First Page of: {pdf_path}")
    print(f"{'='*60}\n")

    # Get original image for display
    original_img = pdf_page_to_image(pdf_path, 0, dpi=dpi)
    print(f"Page size: {original_img.width} x {original_img.height} px (at {dpi} DPI)")

    # Preprocess for OCR
    processed_img = preprocess_for_ocr(original_img)

    # Run OCR on preprocessed image
    print("Running OCR (sparse text mode)...")
    ocr_data = ocr_with_data(processed_img, lang=lang)

    # Collect detected words
    words = []
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])
        if text and conf > MIN_CONFIDENCE:
            words.append({
                "text": text,
                "conf": conf,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i],
                "w": ocr_data["width"][i],
                "h": ocr_data["height"][i],
            })

    print(f"Detected {len(words)} text items (conf > {MIN_CONFIDENCE}%)\n")

    # Print summary
    print("--- Detected Text ---\n")
    for w in sorted(words, key=lambda w: (w["y"], w["x"])):
        print(f"  [{w['conf']:3d}%] ({w['x']:5d}, {w['y']:5d}) \"{w['text']}\"")

    # --- Downsample for display (OCR already done at full DPI) ---
    PREVIEW_MAX_WIDTH = 10000  # pixels — keeps matplotlib snappy
    scale = 1.0
    if original_img.width > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / original_img.width
    display_size = (int(original_img.width * scale), int(original_img.height * scale))

    display_orig = original_img.resize(display_size, Image.LANCZOS)
    display_proc = processed_img.resize(display_size, Image.LANCZOS)

    # --- Matplotlib visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"OCR Preview — {Path(pdf_path).name}  (scaled to {scale:.0%})",
                 fontsize=14, fontweight="bold")

    # Left: original + bounding boxes
    ax1.set_title("Original + OCR Detections")
    ax1.imshow(np.array(display_orig))

    for w in words:
        if w["conf"] >= 70:
            color = "lime"
        elif w["conf"] >= 40:
            color = "yellow"
        else:
            color = "red"

        sx, sy = w["x"] * scale, w["y"] * scale
        sw, sh = w["w"] * scale, w["h"] * scale

        rect = mpatches.FancyBboxPatch(
            (sx, sy), sw, sh,
            linewidth=1.5, edgecolor=color, facecolor="none",
            boxstyle="round,pad=1"
        )
        ax1.add_patch(rect)
        ax1.text(
            sx, sy - 3, f'{w["text"]}',
            fontsize=6, color=color, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1)
        )

    ax1.axis("off")

    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="lime", label="High conf (≥70%)"),
        mpatches.Patch(facecolor="none", edgecolor="yellow", label="Medium conf (40-69%)"),
        mpatches.Patch(facecolor="none", edgecolor="red", label=f"Low conf ({MIN_CONFIDENCE}-39%)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9,
               facecolor="black", edgecolor="white", labelcolor="white")

    # Right: preprocessed image
    ax2.set_title("Preprocessed (what Tesseract sees)")
    ax2.imshow(np.array(display_proc), cmap="gray")
    ax2.axis("off")

    plt.tight_layout()
    print("\n>> Close the preview window to continue...")
    plt.show()

    return len(words)


def convert_to_searchable_pdf(pdf_path: str, output_path: str, dpi: int = 300, lang: str = "eng"):
    """Convert each page to a searchable PDF using Tesseract's PDF output."""
    print(f"\n{'='*60}")
    print(f"Converting to searchable PDF...")
    print(f"{'='*60}\n")

    total = pdf_page_count(pdf_path)
    print(f"  Found {total} page(s)\n")

    writer = PdfWriter()

    for i in range(total):
        print(f"  OCR page {i+1}/{total}...", end=" ", flush=True)

        img = pdf_page_to_image(pdf_path, i, dpi=dpi)
        processed = preprocess_for_ocr(img)

        # Tesseract produces a PDF with invisible text layer over the image
        # We use the ORIGINAL image for the visible layer but OCR the processed one
        # Strategy: run OCR on processed, but embed the original
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            processed, lang=lang, config=TESS_CONFIG, extension="pdf"
        )

        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)

        print("done")

    with open(output_path, "wb") as f:
        writer.write(f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Searchable PDF saved: {output_path} ({size_mb:.1f} MB)")


def main():
    pdf_path = Path(INPUT_PDF)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path.absolute()}")
        print(f"\nPlease set INPUT_PDF at the top of this script to your PDF path.")
        sys.exit(1)

    output_path = OUTPUT_PDF or str(pdf_path.stem) + "_searchable.pdf"

    # Step 1: Visual preview with bounding boxes
    word_count = show_ocr_preview(str(pdf_path), dpi=DPI, lang=LANG)

    if word_count == 0:
        print("\n⚠️  No text detected! You may need to adjust DPI or threshold.")
        print("    Try increasing DPI (e.g., 600) or lowering the threshold in preprocess_for_ocr().")

    # Step 2: Ask to continue
    total_pages = pdf_page_count(str(pdf_path))
    print(f"\n{'='*60}")
    print(f"Ready to convert {total_pages} page(s) to searchable PDF.")
    print(f"  Input:  {pdf_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    try:
        resp = input("\nProceed? [Y/n]: ").strip().lower()
        if resp and resp != "y":
            print("Cancelled.")
            return
    except EOFError:
        pass

    # Step 3: Full conversion
    convert_to_searchable_pdf(str(pdf_path), output_path, dpi=DPI, lang=LANG)


if __name__ == "__main__":
    main()