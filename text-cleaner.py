import pytesseract
import cv2
import numpy as np
from PIL import Image
import re


def is_valid_text(text):
    """Filter out junk detections - keep only meaningful text"""
    text = text.strip()
    if len(text) < 2:
        return False
    # Must contain at least one alphanumeric character
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    # Skip if mostly punctuation/symbols
    alnum_count = sum(1 for c in text if c.isalnum())
    if alnum_count / len(text) < 0.4:
        return False
    return True


def run_tesseract(image, config, total_scale):
    """Run Tesseract on an image and return results"""
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    results = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if not text or conf < 30 or not is_valid_text(text):
            continue
        
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        
        results.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'text': text, 'conf': conf / 100.0
        })
    
    return results


def advanced_schematic_ocr(image_path):
    """Multi-scale OCR using Tesseract for schematics, including rotated text"""
    
    pil_img = Image.open(image_path)
    orig_width, orig_height = pil_img.size
    print(f"Original image size: {orig_width}x{orig_height} pixels")
    
    MAX_DIM = 10000
    
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    max_current_dim = max(img.shape[0], img.shape[1])
    if max_current_dim > MAX_DIM:
        downsample_factor = MAX_DIM / max_current_dim
        print(f"\nDownsampling by {downsample_factor:.2f}x")
        new_width = int(img.shape[1] * downsample_factor)
        new_height = int(img.shape[0] * downsample_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Base image size: {new_width}x{new_height}")
    else:
        downsample_factor = 1.0
    
    all_results = []
    custom_config = r'--oem 1 --psm 11'
    
    for scale in [0.2, 0.3, 0.4, 0.5]:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        
        if width > MAX_DIM or height > MAX_DIM:
            print(f"\nSkipping {scale}x - too large ({width}x{height})")
            continue
        if width < 500 or height < 500:
            print(f"\nSkipping {scale}x - too small ({width}x{height})")
            continue
        
        print(f"\nProcessing at {scale}x ({width}x{height})...")
        
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        total_scale = scale * downsample_factor
        
        dets = run_tesseract(enhanced, custom_config, total_scale)
        print(f"  Found {len(dets)} detections")
        
        for det in dets:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            bbox = [
                [x / total_scale, y / total_scale],
                [(x + w) / total_scale, y / total_scale],
                [(x + w) / total_scale, (y + h) / total_scale],
                [x / total_scale, (y + h) / total_scale]
            ]
            all_results.append((bbox, det['text'], det['conf'], scale))
    
    # Remove duplicates - horizontal detections get priority
    unique_results = remove_duplicates(all_results)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS (deduplicated):")
    print("=" * 70)
    for (bbox, text, conf, scale) in sorted(unique_results, key=lambda x: x[2], reverse=True):
        center = np.mean(bbox, axis=0)
        print(f"{text:40s} | Conf: {conf:.3f} | Scale: {scale}x | Pos: ({center[0]:.0f},{center[1]:.0f})")
    
    return unique_results


def remove_duplicates(results, distance_threshold=50):
    """Remove duplicate detections, keeping highest confidence."""
    sorted_results = sorted(results, key=lambda r: -r[2])
    
    unique = []
    
    for result in sorted_results:
        bbox, text, conf, scale = result
        center = np.mean(bbox, axis=0)
        
        is_duplicate = False
        for idx, (u_bbox, u_text, u_conf, u_scale) in enumerate(unique):
            u_center = np.mean(u_bbox, axis=0)
            dist = np.linalg.norm(center - u_center)
            
            if text.lower().strip() == u_text.lower().strip() and dist < distance_threshold:
                is_duplicate = True
                break
            
            if dist < distance_threshold * 0.5:
                is_duplicate = True
                if conf > u_conf:
                    unique[idx] = result
                break
        
        if not is_duplicate:
            unique.append(result)
    
    return unique


def overlay_results(image_path, results):
    """Draw OCR results on the image and display with matplotlib"""
    pil_img = Image.open(image_path)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    MAX_DISPLAY = 8000
    h, w = img.shape[:2]
    display_scale = 1.0
    if max(h, w) > MAX_DISPLAY:
        display_scale = MAX_DISPLAY / max(h, w)
        img = cv2.resize(img, (int(w * display_scale), int(h * display_scale)), interpolation=cv2.INTER_AREA)
    
    for (bbox, text, conf, scale) in results:
        pts = np.array([[int(x * display_scale), int(y * display_scale)] for x, y in bbox], dtype=np.int32)
        
        # Color by confidence
        if conf >= 0.7:
            color = (0, 200, 0)       # Green
        elif conf >= 0.4:
            color = (0, 200, 200)     # Yellow
        else:
            color = (0, 0, 200)       # Red
        
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        
        label = f"{text} ({conf:.2f})"
        top_left = pts[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        lx = max(top_left[0], 0)
        ly = max(top_left[1] - 5, th + baseline)
        
        cv2.rectangle(img, (lx, ly - th - baseline), (lx + tw, ly + baseline), color, -1)
        cv2.putText(img, label, (lx, ly), font, font_scale, (0, 0, 0), thickness)
    
    import matplotlib.pyplot as plt
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Total text regions shown: {len(results)}")
    
    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    ax.imshow(img_rgb)
    ax.set_axis_off()
    ax.set_title(f'OCR Overlay — {len(results)} detections  |  Green=high conf  Yellow=medium  Red=low')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = 'C1 LV Schematics Copy_2.png'
    Image.MAX_IMAGE_PIXELS = None
    
    results = advanced_schematic_ocr(image_path)
    print(f"\n\nTotal unique text regions detected: {len(results)}")
    
    overlay_results(image_path, results)