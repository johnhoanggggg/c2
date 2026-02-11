#!/usr/bin/env python3
"""
Convert AutoCAD R14 DXF (AC1014) to AutoCAD 2007 DXF (AC1021)
with aggressive cleanup for SolidWorks Electrical compatibility.

Usage:
    # Single file:
    python r14_to_r2007.py input_r14.dxf output_r2007.dxf [--layers layer1,layer2,layer3]
    
    # Folder batch processing:
    python r14_to_r2007.py input_folder/ output_folder/ [--layers layer1,layer2,layer3]
    
Examples:
    python r14_to_r2007.py input.dxf output.dxf --layers 3
    python r14_to_r2007.py ./dxf_files/ ./converted/ --layers 0,3,5
    python r14_to_r2007.py ./dxf_files/ ./converted/
"""

import sys
import os
from pathlib import Path
import ezdxf
from ezdxf.addons import Importer
from ezdxf.math import Vec3


def flatten_entity(entity):
    """Force entity Z to 0 where applicable."""
    try:
        if hasattr(entity.dxf, "elevation"):
            entity.dxf.elevation = 0
        if hasattr(entity.dxf, "start"):
            entity.dxf.start = Vec3(entity.dxf.start.x, entity.dxf.start.y, 0)
        if hasattr(entity.dxf, "end"):
            entity.dxf.end = Vec3(entity.dxf.end.x, entity.dxf.end.y, 0)
        if hasattr(entity.dxf, "center"):
            entity.dxf.center = Vec3(entity.dxf.center.x, entity.dxf.center.y, 0)
        if hasattr(entity.dxf, "insert"):
            entity.dxf.insert = Vec3(entity.dxf.insert.x, entity.dxf.insert.y, 0)
    except Exception:
        pass


def preserve_header_extents(src_doc, dst_doc):
    """Copy drawing extents, limits, and units from source to destination document."""
    header_vars = [
        "$EXTMIN", "$EXTMAX",   # Drawing extents (actual bounding box)
        "$LIMMIN", "$LIMMAX",   # Drawing limits (user-defined paper bounds)
        "$INSBASE",             # Insertion base point
        "$INSUNITS",            # Drawing units
        "$LUNITS",              # Linear units format
        "$LUPREC",              # Linear units precision
        "$AUNITS",              # Angular units format
        "$AUPREC",              # Angular units precision
        "$MEASUREMENT",         # 0=Imperial, 1=Metric
    ]
    
    for var in header_vars:
        try:
            val = src_doc.header[var]
            dst_doc.header[var] = val
            print(f"  Preserved {var} = {val}")
        except (ezdxf.DXFKeyError, KeyError):
            pass


def convert_file(src, dst, layers_to_keep=None):
    """Convert a single DXF file."""
    print(f"\nReading: {src}")
    try:
        doc = ezdxf.readfile(src)
    except Exception as e:
        print(f"ERROR: Failed to read {src}: {e}")
        return False

    # Show layer information
    print("=== LAYER INFORMATION ===")
    layers = list(doc.layers)
    print(f"Total layers: {len(layers)}")
    print("Layer names:", ", ".join([layer.dxf.name for layer in layers]))
    print("=========================")

    # Show original extents
    try:
        ext_min = doc.header["$EXTMIN"]
        ext_max = doc.header["$EXTMAX"]
        print(f"Original extents: ({ext_min[0]:.2f}, {ext_min[1]:.2f}) – ({ext_max[0]:.2f}, {ext_max[1]:.2f})")
        print(f"Original size: {ext_max[0] - ext_min[0]:.2f} × {ext_max[1] - ext_min[1]:.2f}")
    except (ezdxf.DXFKeyError, KeyError, IndexError, TypeError):
        print("Original extents: not defined in header")

    if layers_to_keep:
        print(f"Keeping only layers: {', '.join(layers_to_keep)}")
    else:
        print("Keeping all layers")

    print("Creating new R2007 document")
    new_doc = ezdxf.new(dxfversion="R2007")
    new_msp = new_doc.modelspace()

    importer = Importer(doc, new_doc)
    importer.import_tables(["layers", "linetypes", "styles"])

    print("Importing modelspace entities")
    importer.import_modelspace()

    importer.finalize()

    # Preserve original drawing extents and units
    print("Preserving drawing extents and units...")
    preserve_header_extents(doc, new_doc)

    print("Cleaning entities")
    for e in list(new_msp):
        # Filter by layer if specified
        if layers_to_keep:
            try:
                if e.dxf.layer not in layers_to_keep:
                    new_msp.delete_entity(e)
                    continue
            except Exception:
                new_msp.delete_entity(e)
                continue

        # Drop proxy / unsupported entities
        if e.dxftype() in {
            "PROXY_ENTITY",
            "HATCH",
            "WIPEOUT",
            "IMAGE",
            "OLE2FRAME",
            "OLE2OBJECT",
        }:
            new_msp.delete_entity(e)
            continue

        # Convert SPLINE → polyline
        if e.dxftype() == "SPLINE":
            try:
                points = list(e.approximate(segments=50))
                new_msp.add_lwpolyline(points)
            except Exception:
                pass
            new_msp.delete_entity(e)
            continue

        # Normalize linetype
        try:
            e.dxf.linetype = "CONTINUOUS"
        except Exception:
            pass

        flatten_entity(e)

    print("Exploding blocks")
    for block in list(new_doc.blocks):
        if block.name.startswith("*"):
            continue
        try:
            for ref in new_msp.query(f'INSERT[name=="{block.name}"]'):
                ref.explode()
        except Exception:
            pass

    print(f"Saving as: {dst}")
    try:
        new_doc.saveas(dst)
        print(f"✓ Successfully converted: {src}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save {dst}: {e}")
        return False


def main(src, dst, layers_to_keep=None):
    src_path = Path(src)
    dst_path = Path(dst)

    # Check if source is a directory
    if src_path.is_dir():
        print(f"Batch processing folder: {src}")
        print(f"Output folder: {dst}\n")
        
        # Create output directory if it doesn't exist
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # Find all DXF files in source directory
        dxf_files = list(src_path.glob("*.dxf")) + list(src_path.glob("*.DXF"))
        
        if not dxf_files:
            print(f"No DXF files found in {src}")
            return
        
        print(f"Found {len(dxf_files)} DXF file(s) to convert\n")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        
        for dxf_file in dxf_files:
            output_file = dst_path / dxf_file.name
            if convert_file(str(dxf_file), str(output_file), layers_to_keep):
                success_count += 1
            else:
                fail_count += 1
            print("-" * 60)
        
        print("\n" + "=" * 60)
        print(f"BATCH COMPLETE: {success_count} succeeded, {fail_count} failed")
        print("=" * 60)
    
    # Single file processing
    else:
        convert_file(str(src_path), str(dst_path), layers_to_keep)
        print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    layers = None

    # Parse optional --layers argument
    if len(sys.argv) >= 4 and sys.argv[3] == "--layers":
        if len(sys.argv) >= 5:
            layers = [l.strip() for l in sys.argv[4].split(",")]
        else:
            print("Error: --layers requires a comma-separated list of layer names")
            sys.exit(1)

    main(src_file, dst_file, layers)