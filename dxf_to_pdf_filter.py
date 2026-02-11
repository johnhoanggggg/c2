#!/usr/bin/env python3
"""
Convert DXF to PDF with line segment filtering and OCR text extraction.

Usage:
    python dxf_to_pdf_filter.py input.dxf output.pdf [--min-length 0.5]
    python dxf_to_pdf_filter.py input_folder/ output_folder/ [--min-length 0.5]

The script will:
1. Show a preview window for the first file to adjust filtering
2. Filter out short line segments
3. Perform OCR on the ORIGINAL FULL DXF drawing to extract all text
4. Create a PDF with filtered segments visible and OCR'd text as searchable overlay

Dependencies:
    pip install ezdxf matplotlib pytesseract pillow numpy --break-system-packages
    Also requires Tesseract-OCR installed on system
"""

import sys
import os
from pathlib import Path
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import io

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR will be skipped.")
    print("Install with: pip install pytesseract --break-system-packages")


# ---------------------------------------------------------------
# Helper: compute DXF model-space extents from line/polyline data
# ---------------------------------------------------------------

def get_dxf_extents(doc):
    """Return (x_min, y_min, x_max, y_max) of all LINE / LWPOLYLINE / POLYLINE entities."""
    xs, ys = [], []
    for entity in doc.modelspace():
        try:
            if entity.dxftype() == "LINE":
                xs += [entity.dxf.start.x, entity.dxf.end.x]
                ys += [entity.dxf.start.y, entity.dxf.end.y]
            elif entity.dxftype() == "LWPOLYLINE":
                for pt in entity.get_points():
                    xs.append(pt[0])
                    ys.append(pt[1])
            elif entity.dxftype() == "POLYLINE":
                for v in entity.vertices:
                    xs.append(v.dxf.location.x)
                    ys.append(v.dxf.location.y)
        except Exception:
            continue
    if not xs:
        return 0, 0, 1, 1
    return min(xs), min(ys), max(xs), max(ys)


class DXFFilterConverter:
    def __init__(self, min_length=0.5):
        self.min_length = min_length
        self.accepted = False
        
    def get_line_length(self, entity):
        """Calculate length of a line entity. Returns 0 for non-line entities."""
        try:
            if entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                length = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                return length if length > 0 else 0
            
            elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
                return -1
            
            else:
                return 0
                
        except Exception:
            return 0
    
    def render_dxf_to_image(self, doc, dpi=300):
        """Render the entire DXF document to a high-res image for OCR.
        Returns (image, transform_info) where transform_info helps map pixel coords back to DXF coords.
        """
        try:
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
                       facecolor='white', edgecolor='none',
                       pad_inches=0)
            
            plt.close(fig)
            
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            
            transform_info = {
                'xlim': xlim,
                'ylim': ylim,
                'img_width': img.width,
                'img_height': img.height,
                'dpi': dpi
            }
            
            return img, transform_info
            
        except Exception as e:
            print(f"  Error rendering DXF: {e}")
            return None, None
    
    def perform_ocr_on_dxf(self, doc):
        """Perform OCR on the full rendered DXF document with multiple orientations to detect rotated text."""
        if not TESSERACT_AVAILABLE:
            return [], None
        
        try:
            print("  Rendering DXF for OCR...")
            img, transform_info = self.render_dxf_to_image(doc, dpi=300)
            if img is None:
                return [], None
            
            orig_width = img.width
            orig_height = img.height
            
            config = '--psm 6 -c tessedit_char_whitelist=0123456789.,-+/'
            
            all_text_items = []
            
            rotations = [
                (0, "normal"),
                (90, "90° CCW"),
                (180, "180°"),
                (270, "270° CCW / 90° CW")
            ]
            
            for angle, desc in rotations:
                print(f"  Performing OCR at {desc}...")
                
                if angle == 0:
                    rotated_img = img
                else:
                    rotated_img = img.rotate(angle, expand=True)
                
                ocr_data = pytesseract.image_to_data(rotated_img, config=config, output_type=pytesseract.Output.DICT)
                
                n_boxes = len(ocr_data['text'])
                
                for i in range(n_boxes):
                    text = ocr_data['text'][i].strip()
                    if text:
                        confidence = int(ocr_data['conf'][i])
                        if confidence > 30:
                            x = ocr_data['left'][i]
                            y = ocr_data['top'][i]
                            w = ocr_data['width'][i]
                            h = ocr_data['height'][i]
                            
                            center_x = x + w / 2
                            center_y = y + h / 2
                            
                            orig_w = img.width
                            orig_h = img.height

                            if angle == 0:
                                orig_px_x = center_x
                                orig_px_y = center_y
                            elif angle == 90:
                                orig_px_x = orig_w - center_y
                                orig_px_y = center_x
                            elif angle == 180:
                                orig_px_x = orig_w - center_x
                                orig_px_y = orig_h - center_y
                            elif angle == 270:
                                orig_px_x = center_y
                                orig_px_y = orig_h - center_x
                            
                            all_text_items.append({
                                'text': text,
                                'x': orig_px_x,
                                'y': orig_px_y,
                                'width': w,
                                'height': h,
                                'confidence': confidence,
                                'rotation': angle
                            })
            
            filtered_items = self.deduplicate_ocr_results(all_text_items, transform_info)
            
            print(f"  OCR found {len(filtered_items)} unique text items (from {len(all_text_items)} total detections)")
            return filtered_items, transform_info
            
        except Exception as e:
            print(f"  OCR error: {e}")
            import traceback
            traceback.print_exc()
            return [], None
    
    def deduplicate_ocr_results(self, items, transform_info):
        """Remove duplicate OCR detections from different rotation passes."""
        if not items:
            return []
        
        unique_items = []
        position_threshold = 20  # pixels
        
        for item in items:
            is_duplicate = False
            for existing in unique_items:
                if existing['text'] == item['text']:
                    dx = abs(existing['x'] - item['x'])
                    dy = abs(existing['y'] - item['y'])
                    
                    if dx < position_threshold and dy < position_threshold:
                        if item['confidence'] > existing['confidence']:
                            unique_items.remove(existing)
                            unique_items.append(item)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_items.append(item)
        
        return unique_items
    
    def pixel_to_dxf_coords(self, px_x, px_y, transform_info):
        """Convert pixel coordinates to DXF coordinate space."""
        img_width = transform_info['img_width']
        img_height = transform_info['img_height']
        xlim = transform_info['xlim']
        ylim = transform_info['ylim']
        
        x_min = min(xlim)
        x_max = max(xlim)
        y_min = min(ylim)
        y_max = max(ylim)
        
        norm_x = px_x / img_width
        norm_y = px_y / img_height
        
        dxf_x = x_min + norm_x * (x_max - x_min)
        dxf_y = y_max - norm_y * (y_max - y_min)
        
        return dxf_x, dxf_y
    
    def get_polyline_segments(self, entity):
        """Break a polyline into individual segments with their lengths."""
        segments = []
        try:
            if entity.dxftype() == "LWPOLYLINE":
                points = list(entity.get_points())
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i+1]
                    length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    segments.append({
                        'start': (p1[0], p1[1]),
                        'end': (p2[0], p2[1]),
                        'length': length
                    })
            
            elif entity.dxftype() == "POLYLINE":
                vertices = list(entity.vertices)
                for i in range(len(vertices) - 1):
                    p1 = vertices[i].dxf.location
                    p2 = vertices[i+1].dxf.location
                    length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                    segments.append({
                        'start': (p1.x, p1.y),
                        'end': (p2.x, p2.y),
                        'length': length
                    })
        except Exception:
            pass
        
        return segments
    
    def filter_document(self, doc, min_length):
        """Create a filtered copy of the document."""
        new_doc = ezdxf.new(dxfversion=doc.dxfversion)
        new_msp = new_doc.modelspace()
        
        for layer in doc.layers:
            if layer.dxf.name not in new_doc.layers:
                new_doc.layers.add(layer.dxf.name, color=layer.dxf.color)
        
        msp = doc.modelspace()
        removed_count = 0
        kept_count = 0
        
        for entity in msp:
            length = self.get_line_length(entity)
            
            if entity.dxftype() == "LINE":
                if length > 0 and length >= min_length:
                    new_msp.add_foreign_entity(entity)
                    kept_count += 1
                else:
                    removed_count += 1
            
            elif length == -1:
                segments = self.get_polyline_segments(entity)
                layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
                
                for seg in segments:
                    if seg['length'] >= min_length:
                        new_msp.add_line(
                            start=(seg['start'][0], seg['start'][1], 0),
                            end=(seg['end'][0], seg['end'][1], 0),
                            dxfattribs={'layer': layer}
                        )
                        kept_count += 1
                    else:
                        removed_count += 1
            
            else:
                removed_count += 1
        
        return new_doc, removed_count, kept_count
    
    def show_preview(self, original_doc, dxf_path):
        """Show interactive preview with filter slider."""
        print(f"\nShowing preview for: {dxf_path}")
        print("Adjust the minimum line length slider and click 'Accept' to proceed")
        
        fig = plt.figure(figsize=(14, 10))
        
        ax_main = plt.axes([0.1, 0.25, 0.8, 0.65])
        
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
        slider = Slider(ax_slider, 'Min Length', 0.0, 10.0, valinit=self.min_length, valstep=0.01)
        
        ax_accept = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_cancel = plt.axes([0.81, 0.02, 0.1, 0.04])
        btn_accept = Button(ax_accept, 'Accept', color='lightgreen')
        btn_cancel = Button(ax_cancel, 'Cancel', color='lightcoral')
        
        stats_text = fig.text(0.1, 0.16, '', fontsize=10, family='monospace')
        
        def update_preview(val):
            """Update preview when slider changes."""
            ax_main.clear()
            current_min_length = slider.val
            
            msp = original_doc.modelspace()
            kept_segments = []
            removed_segments = []
            
            for entity in msp:
                length = self.get_line_length(entity)
                
                if entity.dxftype() == "LINE":
                    start = entity.dxf.start
                    end = entity.dxf.end
                    seg = ((start.x, start.y), (end.x, end.y))
                    
                    if length > 0 and length >= current_min_length:
                        kept_segments.append(seg)
                    elif length > 0:
                        removed_segments.append(seg)
                
                elif length == -1:
                    segments = self.get_polyline_segments(entity)
                    for seg_data in segments:
                        seg = (seg_data['start'], seg_data['end'])
                        if seg_data['length'] >= current_min_length:
                            kept_segments.append(seg)
                        else:
                            removed_segments.append(seg)
            
            if TESSERACT_AVAILABLE:
                ocr_items, transform_info = self.perform_ocr_on_dxf(original_doc)
            else:
                ocr_items, transform_info = [], None
            
            for seg in removed_segments:
                ax_main.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                            color='red', linewidth=2, alpha=0.7)
            
            for seg in kept_segments:
                ax_main.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                            color='blue', linewidth=1, alpha=1.0)
            
            if ocr_items and transform_info:
                for item in ocr_items:
                    px_x = item['x'] + item['width'] / 2
                    px_y = item['y'] + item['height'] / 2
                    
                    dxf_x, dxf_y = self.pixel_to_dxf_coords(px_x, px_y, transform_info)
                    
                    ax_main.text(dxf_x, dxf_y, item['text'],
                               fontsize=8, color='red', weight='bold',
                               horizontalalignment='center',
                               verticalalignment='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='red'))
            
            if ocr_items:
                all_text = '\n'.join([item['text'] for item in ocr_items[:20]])
                text_display = f"OCR Found {len(ocr_items)} items:\n{all_text}"
                if len(ocr_items) > 20:
                    text_display += f"\n... and {len(ocr_items) - 20} more"
                
                ax_main.text(0.02, 0.98, text_display, 
                           transform=ax_main.transAxes,
                           fontsize=9, family='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                           color='black')
            elif TESSERACT_AVAILABLE:
                ax_main.text(0.02, 0.98, "OCR: No text detected in drawing", 
                           transform=ax_main.transAxes,
                           fontsize=9, family='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax_main.text(0.02, 0.98, "OCR: pytesseract not installed", 
                           transform=ax_main.transAxes,
                           fontsize=9, family='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
            
            total = len(kept_segments) + len(removed_segments)
            stats_text.set_text(
                f"Minimum Length: {current_min_length:.2f}\n"
                f"Kept: {len(kept_segments)} segments (blue)\n"
                f"Removed: {len(removed_segments)} segments (red)\n"
                f"Total: {total} segments"
            )
            
            kept_patch = mpatches.Patch(color='blue', label='Kept segments')
            removed_patch = mpatches.Patch(color='red', label='Removed segments (filtered out)')
            ax_main.legend(handles=[kept_patch, removed_patch], loc='upper right')
            
            ax_main.set_title(f'DXF Preview - {Path(dxf_path).name}')
            ax_main.set_aspect('equal', adjustable='datalim')
            ax_main.autoscale()
            ax_main.grid(True, alpha=0.3)
            fig.canvas.draw_idle()
        
        def on_accept(event):
            self.min_length = slider.val
            self.accepted = True
            plt.close(fig)
        
        def on_cancel(event):
            self.accepted = False
            plt.close(fig)
        
        slider.on_changed(update_preview)
        btn_accept.on_clicked(on_accept)
        btn_cancel.on_clicked(on_cancel)
        
        update_preview(self.min_length)
        
        plt.show()
        
        return self.accepted
    
    def convert_to_pdf(self, dxf_path, pdf_path, min_length):
        """Convert DXF to PDF preserving original DXF coordinate space and resolution.
        
        The PDF page size is derived from the DXF drawing extents so that
        1 DXF unit maps to 1 PDF point (1/72 inch).  If your DXF is in
        millimetres you'll get a near-physical-size output; adjust
        DXF_UNITS_PER_POINT below if you need a different scale.
        """
        print(f"\nConverting: {dxf_path}")
        print(f"Output: {pdf_path}")
        print(f"Min line length: {min_length}")

        # --- Scale factor: how many DXF units equal one PDF point (1/72 in).
        #     1.0  → 1 DXF unit = 1 pt  (good for mm-based drawings)
        #     25.4/72 ≈ 0.3528 → 1 DXF unit = 1 mm on paper
        #     Change this if your output is too large / too small.
        DXF_UNITS_PER_POINT = 1.0
        
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # ----- Compute drawing extents -----
            x_min, y_min, x_max, y_max = get_dxf_extents(doc)
            dxf_width  = x_max - x_min
            dxf_height = y_max - y_min

            if dxf_width <= 0 or dxf_height <= 0:
                print("  ERROR: Drawing has zero extents")
                return False

            # Add a small margin (2 % each side)
            margin_frac = 0.02
            margin_x = dxf_width  * margin_frac
            margin_y = dxf_height * margin_frac
            
            plot_x_min = x_min - margin_x
            plot_x_max = x_max + margin_x
            plot_y_min = y_min - margin_y
            plot_y_max = y_max + margin_y
            
            plot_width  = plot_x_max - plot_x_min
            plot_height = plot_y_max - plot_y_min

            # Figure size in inches  (72 points per inch)
            fig_w_in = (plot_width  / DXF_UNITS_PER_POINT) / 72.0
            fig_h_in = (plot_height / DXF_UNITS_PER_POINT) / 72.0

            print(f"  DXF extents: ({x_min:.2f}, {y_min:.2f}) – ({x_max:.2f}, {y_max:.2f})")
            print(f"  Plot area:   {plot_width:.2f} × {plot_height:.2f} DXF units")
            print(f"  PDF page:    {fig_w_in:.2f} × {fig_h_in:.2f} inches")

            # ----- OCR on original drawing -----
            ocr_items = []
            transform_info = None
            if TESSERACT_AVAILABLE:
                ocr_items, transform_info = self.perform_ocr_on_dxf(doc)
                if ocr_items:
                    print(f"  OCR extracted: {len(ocr_items)} text items")
                else:
                    print(f"  OCR: No text detected")
            
            # ----- Collect and filter segments -----
            kept_segments = []
            removed_count = 0
            
            for entity in msp:
                length = self.get_line_length(entity)
                
                if entity.dxftype() == "LINE":
                    start = entity.dxf.start
                    end = entity.dxf.end
                    if length > 0 and length >= min_length:
                        kept_segments.append(((start.x, start.y), (end.x, end.y)))
                    else:
                        removed_count += 1
                
                elif length == -1:
                    for seg_data in self.get_polyline_segments(entity):
                        if seg_data['length'] >= min_length:
                            kept_segments.append((seg_data['start'], seg_data['end']))
                        else:
                            removed_count += 1
                else:
                    removed_count += 1
            
            print(f"  Kept: {len(kept_segments)} segments")
            print(f"  Removed: {removed_count} segments")
            
            # ----- Build the PDF figure -----
            fig = plt.figure(figsize=(fig_w_in, fig_h_in))
            # Fill the entire figure — no subplot padding
            ax = fig.add_axes([0, 0, 1, 1])
            
            # Plot kept segments
            for seg in kept_segments:
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 
                       color='black', linewidth=0.5)
            
            # Add OCR text overlay
            if ocr_items and transform_info:
                for item in ocr_items:
                    px_x = item['x'] + item['width'] / 2
                    px_y = item['y'] + item['height'] / 2
                    dxf_x, dxf_y = self.pixel_to_dxf_coords(px_x, px_y, transform_info)
                    ax.text(dxf_x, dxf_y, item['text'],
                           fontsize=8, color='red', weight='bold',
                           horizontalalignment='center',
                           verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                     alpha=0.7, edgecolor='red'))
            
            # Lock axes to exact DXF extents (with margin)
            ax.set_xlim(plot_x_min, plot_x_max)
            ax.set_ylim(plot_y_min, plot_y_max)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Save — no bbox_inches='tight', no extra padding
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=72, pad_inches=0)
            
            plt.close(fig)
            
            print(f"  ✓ Successfully converted with preserved coordinates")
            return True
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    src = sys.argv[1]
    dst = sys.argv[2]
    
    min_length = 0.5
    if len(sys.argv) >= 4 and sys.argv[3] == "--min-length":
        if len(sys.argv) >= 5:
            try:
                min_length = float(sys.argv[4])
            except ValueError:
                print("Error: --min-length must be a number")
                sys.exit(1)
        else:
            print("Error: --min-length requires a value")
            sys.exit(1)
    
    converter = DXFFilterConverter(min_length=min_length)
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Batch processing
    if src_path.is_dir():
        print(f"Batch processing folder: {src}")
        print(f"Output folder: {dst}\n")
        
        dst_path.mkdir(parents=True, exist_ok=True)
        
        dxf_files = list(src_path.glob("*.dxf")) + list(src_path.glob("*.DXF"))
        
        if not dxf_files:
            print(f"No DXF files found in {src}")
            return
        
        print(f"Found {len(dxf_files)} DXF file(s)")
        
        first_file = dxf_files[0]
        print("\n" + "="*60)
        print("PREVIEW MODE - First file")
        print("="*60)
        
        first_doc = ezdxf.readfile(str(first_file))
        if not converter.show_preview(first_doc, str(first_file)):
            print("\nConversion cancelled by user")
            return
        
        print("\n" + "="*60)
        print(f"Processing all files with min_length = {converter.min_length}")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for dxf_file in dxf_files:
            output_file = dst_path / (dxf_file.stem + ".pdf")
            if converter.convert_to_pdf(str(dxf_file), str(output_file), converter.min_length):
                success_count += 1
            else:
                fail_count += 1
            print("-" * 60)
        
        print("\n" + "="*60)
        print(f"BATCH COMPLETE: {success_count} succeeded, {fail_count} failed")
        print("="*60)
    
    # Single file
    else:
        doc = ezdxf.readfile(str(src_path))
        if converter.show_preview(doc, str(src_path)):
            converter.convert_to_pdf(str(src_path), str(dst_path), converter.min_length)
            print("\nDone.")
        else:
            print("\nConversion cancelled by user")


if __name__ == "__main__":
    main()