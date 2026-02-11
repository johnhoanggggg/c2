#!/usr/bin/env python3
"""
Inspect a DXF file to find all entity types and text content.
Usage: python inspect_dxf.py input.dxf
"""

import sys
import ezdxf
from collections import defaultdict

def inspect_dxf(filepath):
    print(f"Inspecting: {filepath}\n")
    
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    
    # Count entity types
    entity_counts = defaultdict(int)
    text_entities = []
    
    for entity in msp:
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        
        # Try to extract any text-like content
        try:
            if entity_type == "TEXT":
                text_entities.append({
                    'type': 'TEXT',
                    'text': entity.dxf.text,
                    'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'unknown'
                })
            elif entity_type == "MTEXT":
                text_entities.append({
                    'type': 'MTEXT',
                    'text': entity.text,
                    'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'unknown'
                })
            elif entity_type == "ATTRIB":
                text_entities.append({
                    'type': 'ATTRIB',
                    'text': entity.dxf.text,
                    'tag': entity.dxf.tag if hasattr(entity.dxf, 'tag') else 'unknown',
                    'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'unknown'
                })
            elif entity_type == "INSERT":
                # Check for attributes in block inserts
                if entity.attribs:
                    for attrib in entity.attribs:
                        text_entities.append({
                            'type': 'INSERT->ATTRIB',
                            'text': attrib.dxf.text,
                            'tag': attrib.dxf.tag if hasattr(attrib.dxf, 'tag') else 'unknown',
                            'block': entity.dxf.name,
                            'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'unknown'
                        })
        except Exception as e:
            pass
    
    # Print entity counts
    print("="*60)
    print("ENTITY TYPES IN MODELSPACE:")
    print("="*60)
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")
    
    # Print text entities
    print("\n" + "="*60)
    print(f"TEXT-CONTAINING ENTITIES: {len(text_entities)}")
    print("="*60)
    
    if text_entities:
        for i, item in enumerate(text_entities[:50], 1):  # Show first 50
            print(f"\n[{i}] Type: {item['type']}")
            if 'tag' in item:
                print(f"    Tag: {item['tag']}")
            if 'block' in item:
                print(f"    Block: {item['block']}")
            print(f"    Layer: {item['layer']}")
            print(f"    Text: {item['text']}")
        
        if len(text_entities) > 50:
            print(f"\n... and {len(text_entities) - 50} more text entities")
    else:
        print("  No text entities found!")
        print("\n  Text might be stored in:")
        print("    - Block definitions (not modelspace)")
        print("    - Attributes within blocks")
        print("    - External references")
        print("    - Other layouts (not modelspace)")
    
    # Check blocks
    print("\n" + "="*60)
    print("BLOCKS IN DOCUMENT:")
    print("="*60)
    
    for block in doc.blocks:
        if not block.name.startswith('*'):
            block_text_count = 0
            for entity in block:
                if entity.dxftype() in ['TEXT', 'MTEXT', 'ATTDEF']:
                    block_text_count += 1
            
            if block_text_count > 0:
                print(f"  {block.name}: {len(block)} entities, {block_text_count} text entities")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_dxf.py input.dxf")
        sys.exit(1)
    
    inspect_dxf(sys.argv[1])