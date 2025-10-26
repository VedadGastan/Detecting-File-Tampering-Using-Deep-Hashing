"""
PDF Data Generation with 15 Forensically-Relevant Tampering Techniques
"""

import fitz
import os
import random
import csv
import shutil
import io
import time
from PIL import Image
from typing import Tuple, Optional
from config import (
    ORIGINAL_PDF_DIR, DATASET_CSV_PATH, TAMPERED_PDF_DIR,
    TAMPER_TYPES, SAMPLES_PER_ORIGINAL
)


# === ALL TAMPERING TECHNIQUES ===
# 1. invisible_text: Invisible Text Injection (ITI) - hidden text off-page
# 2. zero_width_space: Zero-width character injection
# 3. meta_change: Metadata modification (author, dates, producer)
# 4. toc_removal: Table of Contents deletion
# 5. line_artifact: Near-invisible visual artifacts
# 6. image_recompress: Image quality degradation
# 7. font_substitution: Font replacement attack
# 8. page_rotation: Page orientation manipulation
# 9. watermark_injection: Hidden watermark addition
# 10. javascript_injection: Malicious JavaScript embedding
# 11. annotation_injection: Hidden annotation/comment addition
# 12. link_manipulation: URL/hyperlink tampering
# 13. bookmark_manipulation: PDF outline/bookmark changes
# 14. encryption_metadata: Security settings manipulation
# 15. content_stream_reorder: PDF content stream reordering


def extract_page_modalities(pdf_path: str, page_num: int = 0) -> Tuple[str, Image.Image]:
    """Extract text and image from PDF page."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            return "", Image.new('RGB', (224, 224), color='white')
            
        page = doc[page_num]
        text = page.get_text("text")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        doc.close()
        return text, image
    except Exception as e:
        print(f"Extraction error {pdf_path}: {e}")
        return "", Image.new('RGB', (224, 224), color='white')


def _get_page_content_streams(page):
    """Retrieve page content stream references."""
    content_objects = []
    try:
        contents = page.get_contents()
        if contents is None:
            return []
        for item in contents:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                xref = item[0]
            elif isinstance(item, int):
                xref = item
            else:
                continue
            stream = page.parent.xref_stream(xref)
            if stream:
                content_objects.append((xref, stream))
    except:
        pass
    return content_objects


def create_tampered_version(orig_pdf_path: str, tamper_type: str) -> Optional[str]:
    """
    Apply forensically-relevant tampering to PDF.
    
    Severity Levels:
        LOW: Metadata changes, TOC removal, bookmarks
        MEDIUM: Visual artifacts, font changes, watermarks
        HIGH: Invisible text, JavaScript, link manipulation
        CRITICAL: Content stream reordering, encryption tampering
    """
    base_name = os.path.basename(orig_pdf_path)
    tamp_pdf_path = os.path.join(TAMPERED_PDF_DIR, f"tamp_{tamper_type}_{base_name}")
    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    temp_path = tamp_pdf_path + ".tmp"
    doc = None
    
    try:
        doc = fitz.open(orig_pdf_path)
        if doc.page_count == 0:
            doc.close()
            return None
        page = doc[0]

        # === HIGH SEVERITY ATTACKS ===
        if tamper_type == "invisible_text":
            invisible_text_op = b" BT\n1 0 0 1 -1000 -1000 Tm\n/F1 1 Tf\n(HIDDEN_MARKER) Tj\nET\n"
            content_streams = _get_page_content_streams(page)
            if content_streams:
                xref, original_content = content_streams[0]
                modified_content = original_content.rstrip(b' \n') + b'\n' + invisible_text_op
                doc.update_stream(xref, modified_content)
            else:
                rect = fitz.Rect(-1000, -1000, -999, -999)
                page.insert_textbox(rect, "HIDDEN_MARKER", fontsize=1, color=(1, 1, 1))

        elif tamper_type == "zero_width_space":
            rect = fitz.Rect(1, 1, 2, 2)
            page.insert_textbox(rect, "\u200b" * 15, fontsize=0.1, color=(1, 1, 1), render_mode=3)

        elif tamper_type == "javascript_injection":
            js_code = "app.alert('Document verification required');"
            doc.embeddedFileAdd("script.js", js_code.encode(), filename="verification.js")

        elif tamper_type == "link_manipulation":
            links = page.get_links()
            if links:
                link = links[0]
                link['uri'] = "https://malicious-site.com/verify"
                page.update_link(link)
            else:
                rect = fitz.Rect(50, 50, 150, 70)
                page.insert_link({'kind': 2, 'from': rect, 'uri': 'https://phishing.com'})

        # === MEDIUM SEVERITY ATTACKS ===
        elif tamper_type == "font_substitution":
            text_instances = page.get_text("dict")
            if text_instances and "blocks" in text_instances:
                try:
                    rect = fitz.Rect(10, 10, 200, 30)
                    page.insert_textbox(rect, "Substituted Font Text", fontname="Times-Roman", fontsize=12)
                except:
                    page.draw_rect(fitz.Rect(5, 5, 10, 10), color=(0.95, 0.95, 0.95))

        elif tamper_type == "page_rotation":
            page.set_rotation(90)

        elif tamper_type == "watermark_injection":
            rect = fitz.Rect(page.rect.width/2 - 100, page.rect.height/2 - 20,
                           page.rect.width/2 + 100, page.rect.height/2 + 20)
            page.insert_textbox(rect, "CONFIDENTIAL", fontsize=48, 
                              color=(0.9, 0.9, 0.9), rotate=45, overlay=False)

        elif tamper_type == "line_artifact":
            page.draw_line(fitz.Point(5, 5), fitz.Point(15, 5), 
                         color=(0.98, 0.98, 0.98), width=0.5, overlay=True)

        elif tamper_type == "image_recompress":
            try:
                pix = page.get_pixmap(alpha=False)
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                output = io.BytesIO()
                pil_img.save(output, format="JPEG", quality=60)
                output.seek(0)
                page.clean_contents()
                page.insert_image(page.rect, stream=output.getvalue())
            except:
                page.draw_rect(fitz.Rect(0, 0, 3, 3), color=(0.99, 0.99, 0.99))

        elif tamper_type == "annotation_injection":
            annot = page.add_text_annot(fitz.Point(50, 50), "Suspicious annotation")
            annot.set_opacity(0.1)

        # === LOW SEVERITY ATTACKS ===
        elif tamper_type == "meta_change":
            doc.set_metadata({
                "author": f"Attacker_{random.randint(1000, 9999)}",
                "producer": "Unknown Producer v2.0",
                "creationDate": "D:20000101120000Z",
                "modDate": "D:20241225120000Z"
            })

        elif tamper_type == "toc_removal":
            doc.set_toc([])

        elif tamper_type == "bookmark_manipulation":
            fake_toc = [
                [1, "Fake Section 1", 1],
                [2, "Malicious Link", 1, {"kind": 2, "uri": "https://fake.com"}]
            ]
            doc.set_toc(fake_toc)

        elif tamper_type == "encryption_metadata":
            doc.set_metadata({
                "encryption": "Modified",
                "security": "Bypassed"
            })

        # === CRITICAL SEVERITY ATTACKS ===
        elif tamper_type == "content_stream_reorder":
            content_streams = _get_page_content_streams(page)
            if len(content_streams) > 1:
                xref1, stream1 = content_streams[0]
                xref2, stream2 = content_streams[1]
                doc.update_stream(xref1, stream2)
                doc.update_stream(xref2, stream1)
            else:
                page.draw_rect(fitz.Rect(0, 0, 2, 2), color=(0.98, 0.98, 0.98))

        else:
            print(f"Unknown tamper type: {tamper_type}")
            doc.close()
            return None
        
        doc.save(temp_path, garbage=4, deflate=True, clean=True)
        doc.close()
        
        if os.path.exists(tamp_pdf_path):
            os.remove(tamp_pdf_path)
        shutil.move(temp_path, tamp_pdf_path)
        
        return tamp_pdf_path

    except Exception as e:
        print(f"Tampering error [{tamper_type}] on {base_name}: {e}")
        if doc:
            try:
                doc.close()
            except:
                pass
        for path in [temp_path, tamp_pdf_path]:
            if os.path.exists(path):
                for _ in range(5):
                    try:
                        os.remove(path)
                        break
                    except:
                        time.sleep(0.1)
        return None


# Severity mapping for analysis
SEVERITY_MAP = {
    "invisible_text": "HIGH",
    "zero_width_space": "HIGH",
    "javascript_injection": "HIGH",
    "link_manipulation": "HIGH",
    "font_substitution": "MEDIUM",
    "page_rotation": "MEDIUM",
    "watermark_injection": "MEDIUM",
    "line_artifact": "MEDIUM",
    "image_recompress": "MEDIUM",
    "annotation_injection": "MEDIUM",
    "meta_change": "LOW",
    "toc_removal": "LOW",
    "bookmark_manipulation": "LOW",
    "encryption_metadata": "LOW",
    "content_stream_reorder": "CRITICAL"
}


def generate_dataset_csv(num_samples_per_orig: int = SAMPLES_PER_ORIGINAL) -> None:
    """Generate dataset with identity pairs (label=0) and tampered pairs (label=1)."""
    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("ERROR: No PDF files in ORIGINAL_PDF_DIR")
        return
    
    print(f"Found {len(pdf_files)} PDFs")
    data_log = []
    success = 0
    failed = 0

    for orig_file in pdf_files:
        orig_path = os.path.join(ORIGINAL_PDF_DIR, orig_file)
        data_log.append([orig_path, orig_path, 0, "identity", "NONE"])

        for _ in range(num_samples_per_orig):
            t_type = random.choice(TAMPER_TYPES)
            tamp_path = create_tampered_version(orig_path, t_type)
            
            if tamp_path:
                severity = SEVERITY_MAP.get(t_type, "UNKNOWN")
                data_log.append([orig_path, tamp_path, 1, t_type, severity])
                success += 1
            else:
                failed += 1

    with open(DATASET_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original_Path', 'Paired_Path', 'Label', 'Tamper_Type', 'Severity'])
        writer.writerows(data_log)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {len(data_log)} pairs")
    print(f"Identity: {len(pdf_files)} | Tampered: {success} | Failed: {failed}")
    print(f"Saved: {DATASET_CSV_PATH}")
    print(f"{'='*50}\n")