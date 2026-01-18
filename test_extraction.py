
import os
from docx import Document
from src.data_loader import get_data_path

def inspect_docx():
    file_path = get_data_path('Laporan', 'UNIT 1', '400 V', 'C3WP1A_000.docx')
    doc = Document(file_path)
    
    print("Inspecting relationships...")
    found = False
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            blob = rel.target_part.blob
            print(f"Found image: {rel.target_ref}, Size: {len(blob)} bytes")
            found = True
            
    if not found:
        print("No images found in document part relationships.")

if __name__ == "__main__":
    inspect_docx()
