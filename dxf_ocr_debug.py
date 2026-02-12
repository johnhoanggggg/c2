#!/usr/bin/env python3
"""
DXF OCR Debug Tool
==================
Iterates through a folder of DXF files, extracts a target layer (default: layer "4"),
renders the drawing, performs OCR using the same approach as the wires pipeline
(--psm 11, digits-only, multi-rotation, high DPI), and displays an interactive
matplotlib window with OCR results overlaid on the original rendered page.

Usage:
    python dxf_ocr_debug.py <dxf_folder> [--layer 4] [--dpi 600] [--min-conf 20]
    python dxf_ocr_debug.py single_file.dxf --layer 4

Controls:
    Right arrow / N  : Next file
    Left arrow / P   : Previous file
    Q / Escape       : Quit
    C                : Toggle confidence coloring
    T                : Toggle text labels
    +/-              : Increase/decrease min confidence threshold
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ezdxf
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("ERROR: pytesseract not installed. Install with: pip install pytesseract")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------

def extract_layer(src_path: str, layer_name: str) -> Optional[ezdxf.document.Drawing]:
    """Read a DXF file and return a filtered document containing only entities on the
    specified layer. Returns None if the file can't be read or the layer is empty."""
    try:
        doc = ezdxf.readfile(src_path)
    except Exception as e:
        print(f"  ERROR reading {src_path}: {e}")
        return None

    layers = [l.dxf.name for l in doc.layers]
    if layer_name not in layers:
        print(f"  Layer '{layer_name}' not found. Available: {', '.join(layers)}")
        return None

    # Keep the original document tables/styles/fonts and only filter modelspace
    # entities. This preserves CAD style context better than rebuilding a new doc.
    msp = doc.modelspace()
    count = 0
    to_delete = []
    for entity in msp:
        try:
            if entity.dxf.layer == layer_name:
                count += 1
            else:
                to_delete.append(entity)
        except Exception:
            pass

    if count == 0:
        print(f"  Layer '{layer_name}' exists but has 0 entities")
        return None

    for entity in to_delete:
        try:
            msp.delete_entity(entity)
        except Exception:
            pass

    print(f"  Extracted {count} entities from layer '{layer_name}'")
    return doc


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_dxf_to_image(doc, dpi: int = 600) -> Tuple[Optional[Image.Image], Optional[Dict]]:
    """Render DXF modelspace to a PIL image. Returns (image, transform_info)."""
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    fig = plt.figure(figsize=(11, 8.5), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')

    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi,
                facecolor='white', edgecolor='none', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    transform_info = {
        'xlim': xlim,
        'ylim': ylim,
        'img_width': img.width,
        'img_height': img.height,
        'dpi': dpi,
    }
    return img, transform_info


def pixel_to_dxf_coords(px_x, px_y, tinfo):
    """Convert pixel coordinates to DXF coordinate space."""
    img_w = tinfo['img_width']
    img_h = tinfo['img_height']
    x_min, x_max = min(tinfo['xlim']), max(tinfo['xlim'])
    y_min, y_max = min(tinfo['ylim']), max(tinfo['ylim'])

    norm_x = px_x / img_w
    norm_y = px_y / img_h

    dxf_x = x_min + norm_x * (x_max - x_min)
    dxf_y = y_max - norm_y * (y_max - y_min)
    return dxf_x, dxf_y


# ---------------------------------------------------------------------------
# OCR (wires-style: --psm 11, digits, multi-rotation)
# ---------------------------------------------------------------------------

def preprocess_for_ocr(img: Image.Image, binarize: bool = True) -> Image.Image:
    """Preprocess image before OCR. Uses Otsu binarization when enabled."""
    gray = np.array(img.convert("L"))
    if not binarize:
        return Image.fromarray(gray, mode="L").convert("RGB")

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    w_bg = 0.0
    best_var = -1.0
    threshold = 127
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            threshold = t

    bw = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(bw, mode="L").convert("RGB")


def perform_ocr(img: Image.Image, tinfo: Dict, dpi: int = 600,
                min_confidence: int = 20, binarize: bool = True) -> List[Dict]:
    """Run OCR at multiple rotations using the wires pipeline approach.
    Returns list of dicts: text, dxf_x, dxf_y, px_x, px_y, confidence, rotation, width, height."""

    # Wires-style config: sparse text, digits only
    config = '--psm 11 -c tessedit_char_whitelist=+ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

    ocr_img = preprocess_for_ocr(img, binarize=binarize)
    all_items = []
    rotations = [(0, "0deg")]

    for angle, desc in rotations:
        rotated = ocr_img if angle == 0 else ocr_img.rotate(angle, expand=True)
        print(f"    OCR input [{desc}]: {rotated.width}x{rotated.height} px (render_dpi={dpi})")
        ocr_data = pytesseract.image_to_data(
            rotated, config=config, output_type=pytesseract.Output.DICT
        )

        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if not text:
                continue
            confidence = int(ocr_data['conf'][i])
            if confidence < min_confidence:
                continue

            cx = ocr_data['left'][i] + ocr_data['width'][i] / 2
            cy = ocr_data['top'][i] + ocr_data['height'][i] / 2
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            # Map rotated pixel coords back to original image coords
            orig_w, orig_h = ocr_img.width, ocr_img.height
            if angle == 0:
                opx, opy = cx, cy
            elif angle == 90:
                opx, opy = orig_w - cy, cx
            elif angle == 270:
                opx, opy = cy, orig_h - cx
            else:
                opx, opy = cx, cy

            dxf_x, dxf_y = pixel_to_dxf_coords(opx, opy, tinfo)

            all_items.append({
                'text': text,
                'dxf_x': dxf_x,
                'dxf_y': dxf_y,
                'px_x': opx,
                'px_y': opy,
                'confidence': confidence,
                'rotation': angle,
                'width': w,
                'height': h,
            })

    # Deduplicate (wires-style: position threshold in DXF units)
    unique = []
    pos_thresh = 0.001  # DXF units
    for item in all_items:
        dup = False
        for ex in unique:
            if ex['text'] == item['text']:
                if (abs(ex['dxf_x'] - item['dxf_x']) < pos_thresh and
                        abs(ex['dxf_y'] - item['dxf_y']) < pos_thresh):
                    if item['confidence'] > ex['confidence']:
                        unique.remove(ex)
                        unique.append(item)
                    dup = True
                    break
        if not dup:
            unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

class OCRDebugViewer:
    """Interactive matplotlib viewer that overlays OCR results on the DXF page."""

    def __init__(self, dxf_files: List[Path], layer: str, dpi: int, min_conf: int,
                 display_dpi: int, anno_scale: float, binarize: bool):
        self.dxf_files = dxf_files
        self.layer = layer
        self.dpi = dpi
        self.min_conf = min_conf
        self.display_dpi = display_dpi
        self.anno_scale = anno_scale
        self.binarize = binarize
        self.idx = 0
        self.show_text = True
        self.show_confidence_color = True
        self.show_ocr_input = False

        # Cache for processed results
        self.cache: Dict[int, dict] = {}

        self.fig = None
        self.ax_main = None

    def process_file(self, idx: int) -> dict:
        """Process a single DXF file and return rendering + OCR results."""
        if idx in self.cache:
            return self.cache[idx]

        path = self.dxf_files[idx]
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(self.dxf_files)}] {path.name}")
        print(f"{'='*60}")

        result = {
            'path': path,
            'layer_img': None,
            'ocr_input_img': None,
            'tinfo': None,
            'ocr_items': [],
            'error': None,
        }

        # Extract target layer
        layer_doc = extract_layer(str(path), self.layer)
        if layer_doc is None:
            result['error'] = f"Could not extract layer '{self.layer}'"
            self.cache[idx] = result
            return result

        # Render layer for OCR at high DPI
        print(f"  Rendering layer '{self.layer}' at {self.dpi} DPI for OCR...")
        layer_img, tinfo = render_dxf_to_image(layer_doc, dpi=self.dpi)
        result['layer_img'] = layer_img
        result['ocr_input_img'] = preprocess_for_ocr(layer_img, binarize=self.binarize) if layer_img else None
        result['tinfo'] = tinfo

        # Perform OCR
        if layer_img is not None and tinfo is not None:
            print(f"  Running OCR (wires-style, psm=11, digits, {self.dpi} DPI)...")
            result['ocr_items'] = perform_ocr(
                layer_img, tinfo, dpi=self.dpi, min_confidence=self.min_conf, binarize=self.binarize
            )
            print(f"  Found {len(result['ocr_items'])} OCR detections")

            # Print summary
            for item in sorted(result['ocr_items'], key=lambda x: -x['confidence']):
                print(f"    '{item['text']}' conf={item['confidence']} "
                      f"rot={item['rotation']}deg "
                      f"pos=({item['dxf_x']:.3f}, {item['dxf_y']:.3f})")

        self.cache[idx] = result
        return result

    def confidence_color(self, conf: int) -> str:
        if conf >= 70:
            return 'lime'
        elif conf >= 40:
            return 'yellow'
        else:
            return 'red'

    def draw(self):
        """Draw the current file's visualization."""
        result = self.process_file(self.idx)
        path = result['path']

        self.ax_main.clear()

        # Title
        status = f"[{self.idx+1}/{len(self.dxf_files)}] {path.name}"
        if result['error']:
            status += f"  |  ERROR: {result['error']}"
        else:
            n = len(result['ocr_items'])
            status += f"  |  Layer '{self.layer}'  |  {n} OCR detections  |  min_conf={self.min_conf}"
        self.ax_main.set_title(status, fontsize=10, fontfamily='monospace')

        if self.show_ocr_input and result['layer_img'] is not None:
            # Show the exact image passed to Tesseract (0deg pass), unscaled.
            ocr_input = result['ocr_input_img'] if result['ocr_input_img'] is not None else result['layer_img']
            layer_arr = np.array(ocr_input)
            self.ax_main.imshow(layer_arr, aspect='equal', alpha=1.0)
            status += (f"  |  OCR INPUT VIEW: {ocr_input.width}x"
                       f"{ocr_input.height}px @ {self.dpi} DPI"
                       f"  |  binarize={'ON' if self.binarize else 'OFF'}")
            self.ax_main.set_title(status, fontsize=10, fontfamily='monospace')
        else:
            # Show only the selected layer.
            if result['layer_img'] is not None:
                layer_arr = np.array(result['layer_img'])
                self.ax_main.imshow(layer_arr, aspect='equal', alpha=1.0)

        # Overlay OCR results
        if result['ocr_items'] and result['layer_img'] is not None:
            tinfo = result['tinfo']
            img_w = result['layer_img'].width
            img_h = result['layer_img'].height

            # No cross-image resize: OCR coordinates are in the same pixel space as layer_img.
            sx, sy = 1.0, 1.0

            # Keep overlays readable at high OCR DPI; user multiplier can tune it.
            dpi_scale = max(0.2, min(1.0, 300.0 / max(1, self.dpi)))
            geom_scale = max(0.15, min(1.0, min(sx, sy)))
            anno_scale = max(0.05, self.anno_scale * dpi_scale * geom_scale)
            line_w = max(0.5, 1.3 * anno_scale)
            font_sz = max(4.0, 8.0 * anno_scale)
            y_offset = max(2.0, 4.0 * anno_scale)
            pad = max(0.02, 0.10 * anno_scale)

            for item in result['ocr_items']:
                if item['confidence'] < self.min_conf:
                    continue

                px = item['px_x'] * sx
                py = item['px_y'] * sy
                hw = item['width'] * sx / 2
                hh = item['height'] * sy / 2

                color = self.confidence_color(item['confidence']) if self.show_confidence_color else 'cyan'

                # Bounding box
                rect = patches.Rectangle(
                    (px - hw, py - hh), hw * 2, hh * 2,
                    linewidth=line_w, edgecolor=color, facecolor='none', alpha=0.8
                )
                self.ax_main.add_patch(rect)

                # Text label
                if self.show_text:
                    rot_label = f" @{item['rotation']}d" if item['rotation'] != 0 else ""
                    label = f"{item['text']} ({item['confidence']}%{rot_label})"
                    self.ax_main.text(
                        px - hw, py - hh - y_offset, label,
                        fontsize=font_sz, color='white', fontfamily='monospace',
                        bbox=dict(facecolor=color, alpha=0.75, edgecolor='none',
                                  boxstyle=f'round,pad={pad}'),
                        verticalalignment='bottom',
                    )

        self.ax_main.axis('off')

        # Legend
        legend_text = (
            "Nav: Left/Right or N/P  |  Q: Quit  |  "
            "T: Toggle text  |  C: Toggle color  |  I: OCR input view  |  [ ]: Label size  |  +/-: Confidence threshold"
        )
        self.fig.text(0.5, 0.01, legend_text, ha='center', fontsize=8,
                      fontfamily='monospace', color='gray')

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ('right', 'n'):
            if self.idx < len(self.dxf_files) - 1:
                self.idx += 1
                self.draw()
        elif event.key in ('left', 'p'):
            if self.idx > 0:
                self.idx -= 1
                self.draw()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)
        elif event.key == 't':
            self.show_text = not self.show_text
            self.draw()
        elif event.key == 'c':
            self.show_confidence_color = not self.show_confidence_color
            self.draw()
        elif event.key == 'i':
            self.show_ocr_input = not self.show_ocr_input
            mode = "ON" if self.show_ocr_input else "OFF"
            print(f"  OCR input view: {mode}")
            self.draw()
        elif event.key == ']':
            self.anno_scale = min(4.0, self.anno_scale * 1.2)
            print(f"  Annotation scale: {self.anno_scale:.2f}")
            self.draw()
        elif event.key == '[':
            self.anno_scale = max(0.05, self.anno_scale / 1.2)
            print(f"  Annotation scale: {self.anno_scale:.2f}")
            self.draw()
        elif event.key == '+' or event.key == '=':
            self.min_conf = min(95, self.min_conf + 5)
            print(f"  Min confidence: {self.min_conf}")
            self.draw()
        elif event.key == '-':
            self.min_conf = max(0, self.min_conf - 5)
            print(f"  Min confidence: {self.min_conf}")
            self.draw()

    def run(self):
        self.fig = plt.figure(figsize=(18, 12), dpi=self.display_dpi)
        self.ax_main = self.fig.add_axes([0.02, 0.04, 0.96, 0.92])
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw()
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DXF OCR Debug Tool - overlay OCR results on DXF layer rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="DXF file or folder of DXF files")
    parser.add_argument("--layer", default="4",
                        help="Layer name to extract (default: 4)")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="DPI for OCR rendering (default: 300)")
    parser.add_argument("--min-conf", type=int, default=20,
                        help="Minimum OCR confidence threshold (default: 20)")
    parser.add_argument("--display-dpi", type=int, default=600,
                        help="Display DPI for the viewer canvas (default: 600)")
    parser.add_argument("--anno-scale", type=float, default=1.0,
                        help="Overlay annotation size multiplier (default: 1.0; lower = smaller labels)")
    parser.add_argument("--no-binarize", action="store_true",
                        help="Disable OCR binarization preprocessing (enabled by default)")
    args = parser.parse_args()

    source = Path(args.input)
    if source.is_dir():
        dxf_files = sorted(source.glob("*.dxf")) + sorted(source.glob("*.DXF"))
        if not dxf_files:
            print(f"No DXF files found in {source}")
            sys.exit(1)
        print(f"Found {len(dxf_files)} DXF file(s) in {source}")
    elif source.is_file():
        dxf_files = [source]
    else:
        print(f"Path not found: {source}")
        sys.exit(1)

    viewer = OCRDebugViewer(
        dxf_files, args.layer, args.dpi, args.min_conf, args.display_dpi, args.anno_scale,
        binarize=not args.no_binarize
    )
    viewer.run()


if __name__ == "__main__":
    main()
