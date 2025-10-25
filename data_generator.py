"""
PDF Data Generation and Tampering Module
Implements various tampering techniques for forensics research
"""

import fitz  # PyMuPDF
import os
import random
import csv
import shutil
import io
import time
from PIL import Image
from typing import Tuple, Optional, List
from config import (
    ORIGINAL_PDF_DIR, DATASET_CSV_PATH, TAMPERED_PDF_DIR,
    TAMPER_TYPES, SAMPLES_PER_ORIGINAL
)


def extract_page_modalities(pdf_path: str, page_num: int = 0) -> Tuple[str, Image.Image]:
    """
    Extracts raw text and high-resolution image for a given PDF page.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to extract (default: 0)
        
    Returns:
        Tuple of (text_content, PIL_Image)
    """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            raise ValueError(f"PDF {pdf_path} has no pages")
            
        page = doc[page_num]
        text = page.get_text("text")
        
        # Extract high-resolution image (2x scaling)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        doc.close()
        
        return text, image
    except Exception as e:
        print(f"Error extracting modalities from {pdf_path}: {e}")
        # Return empty data as fallback
        return "", Image.new('RGB', (224, 224), color='white')


def _get_page_content_streams(page) -> List[Tuple[int, bytes]]:
    """
    Retrieves the cross-reference (xref) and content of a page's stream objects.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        List of tuples containing (xref, stream_content)
    """
    content_objects = []
    try:
        contents = page.get_contents()
        if contents is None:
            return []

        for item in contents:
            # Handle different content list formats
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                xref = item[0]
            elif isinstance(item, int):
                xref = item
            else:
                continue
                
            # Use the correct PyMuPDF method
            stream = page.parent.xref_stream(xref)
            
            if stream:
                content_objects.append((xref, stream))
    except Exception as e:
        print(f"Warning: Could not extract content streams: {e}")
        
    return content_objects


def create_tampered_version(orig_pdf_path: str, tamper_type: str) -> Optional[str]:
    """
    Creates a tampered version of a PDF using various forensics-relevant techniques.
    
    Tamper Types:
        - invisible_text: Invisible Text Injection (ITI)
        - zero_width_space: Zero-width character injection
        - meta_change: Metadata modification
        - toc_removal: Table of Contents removal
        - line_artifact: Near-invisible visual artifact
        - image_recompress: Image quality reduction
    
    Args:
        orig_pdf_path: Path to the original PDF
        tamper_type: Type of tampering to apply
        
    Returns:
        Path to tampered PDF if successful, None otherwise
    """
    base_name = os.path.basename(orig_pdf_path)
    tamp_pdf_path = os.path.join(TAMPERED_PDF_DIR, f"tamp_{tamper_type}_{base_name}")

    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    temp_path = tamp_pdf_path + ".tmp"
    doc = None
    
    try:
        # Step 1: Create a NEW document instead of copying
        doc = fitz.open(orig_pdf_path)
        
        if doc.page_count == 0:
            print(f"Skipping empty document: {orig_pdf_path}")
            doc.close()
            return None
        
        # Step 2: Apply tampering to the first page
        page = doc[0]

        if tamper_type == "invisible_text":
            # Invisible Text Injection (ITI) - Primary forensics technique
            invisible_text_op = b" BT\n1 0 0 1 -1000 -1000 Tm\n/F1 1 Tf\n(ITI_MARKER) Tj\nET\n"
            content_streams = _get_page_content_streams(page)
            if content_streams:
                xref, original_content = content_streams[0]
                modified_content = original_content.rstrip(b' \n') + b'\n' + invisible_text_op + b' \n'
                doc.update_stream(xref, modified_content)
            else:
                # Fallback: Use textbox method
                rect = fitz.Rect(-1000, -1000, -999, -999)
                page.insert_textbox(rect, "ITI_MARKER", fontsize=1, color=(1, 1, 1))

        elif tamper_type == "zero_width_space":
            # Zero-width space injection (subtle text-layer tampering)
            rect = fitz.Rect(1, 1, 2, 2)
            page.insert_textbox(
                rect, 
                "\u200b" * 10,  # Zero-width spaces
                fontsize=0.1, 
                color=(1, 1, 1), 
                render_mode=3
            )
            
        elif tamper_type == "meta_change":
            # Metadata modification (common in document fraud)
            doc.set_metadata({
                "author": f"System_{random.randint(1000, 9999)}",
                "producer": "Forensics Test Engine v1.0",
                "creationDate": "D:19990101120000Z",
                "modDate": "D:20241001120000Z"
            })

        elif tamper_type == "toc_removal":
            # Remove Table of Contents (structural modification)
            doc.set_toc([])

        elif tamper_type == "line_artifact":
            # Near-invisible visual artifact (subtle visual tampering)
            page.draw_line(
                p1=fitz.Point(5, 5), 
                p2=fitz.Point(15, 5), 
                color=(0.98, 0.98, 0.98), 
                width=0.5, 
                overlay=True
            )
        
        elif tamper_type == "image_recompress":
            # Image recompression (degrades image quality)
            try:
                pix = page.get_pixmap(alpha=False)
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                output = io.BytesIO()
                pil_img.save(output, format="JPEG", quality=85)
                output.seek(0)
                
                page.clean_contents()
                page.insert_image(page.rect, stream=output.getvalue())
            except Exception as img_err:
                # Fallback: just add a small artifact instead
                page.draw_rect(fitz.Rect(0, 0, 5, 5), color=(0.99, 0.99, 0.99), width=0.1)
            
        else:
            print(f"Unknown tamper type: {tamper_type}")
            doc.close()
            return None
        
        # Step 3: Save to a temporary file first
        doc.save(
            temp_path,
            garbage=4,  # Maximum garbage collection
            deflate=True,  # Compress streams
            clean=True  # Clean up
        )
        doc.close()
        
        # Step 4: Move temp file to final destination
        if os.path.exists(tamp_pdf_path):
            os.remove(tamp_pdf_path)
        shutil.move(temp_path, tamp_pdf_path)
        
        return tamp_pdf_path

    except Exception as e:
        print(f"Error during PDF modification of {base_name} (type: {tamper_type}): {e}")
        
        # Cleanup
        if doc:
            try:
                doc.close()
            except:
                pass
        
        # Remove temporary files with retry logic
        for path in [temp_path, tamp_pdf_path]:
            if os.path.exists(path):
                for attempt in range(5):
                    try:
                        os.remove(path)
                        break
                    except Exception:
                        time.sleep(0.1)
                        
        return None


def generate_dataset_csv(num_samples_per_orig: int = SAMPLES_PER_ORIGINAL) -> None:
    """
    Generates the dataset CSV file with paired original-tampered PDFs.
    
    Creates:
        - Identity pairs (original-original, label=0)
        - Tampered pairs (original-tampered, label=1)
    
    Args:
        num_samples_per_orig: Number of tampered versions per original PDF
    """
    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("ERROR: No PDF files found in ORIGINAL_PDF_DIR")
        return
    
    print(f"Found {len(pdf_files)} original PDF files")
    data_log = []
    successful_tamperings = 0
    failed_tamperings = 0

    for orig_file in pdf_files:
        orig_path = os.path.join(ORIGINAL_PDF_DIR, orig_file)
        
        # 1. SIMILAR pair (Original vs. Original) -> Label 0
        data_log.append([orig_path, orig_path, 0, "identity"])

        # 2. DISSIMILAR pairs (Original vs. Tampered) -> Label 1
        for i in range(num_samples_per_orig):
            t_type = random.choice(TAMPER_TYPES)
            tamp_path = create_tampered_version(orig_path, t_type)
            
            if tamp_path:
                data_log.append([orig_path, tamp_path, 1, t_type])
                successful_tamperings += 1
            else:
                failed_tamperings += 1

    # Save the dataset log
    with open(DATASET_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original_Path', 'Paired_Path', 'Label', 'Tamper_Type'])
        writer.writerows(data_log)
    
    print(f"\n{'='*60}")
    print(f"Dataset Generation Summary:")
    print(f"  Total pairs created: {len(data_log)}")
    print(f"  Identity pairs: {len(pdf_files)}")
    print(f"  Successful tamperings: {successful_tamperings}")
    print(f"  Failed tamperings: {failed_tamperings}")
    print(f"  Dataset saved to: {DATASET_CSV_PATH}")
    print(f"{'='*60}\n")