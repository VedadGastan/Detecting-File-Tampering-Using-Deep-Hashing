import fitz
import os
import random
import csv
import shutil
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
from tqdm import tqdm
from config import (
    ORIGINAL_PDF_DIR, DATASET_CSV_PATH, TAMPERED_PDF_DIR,
    TAMPER_TYPES, SAMPLES_PER_ORIGINAL
)


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
    """Retrieve page content stream references with safety checks."""
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
            try:
                stream = page.parent.xref_stream(xref)
                if stream:
                    content_objects.append((xref, stream))
            except:
                continue
    except:
        pass
    return content_objects


def _create_patch_image_stream() -> io.BytesIO:
    """Creates a '999' patch image in memory."""
    try:
        patch_img = Image.new('RGB', (100, 50), 'white')
        draw = ImageDraw.Draw(patch_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((30, 15), "999", fill='black', font=font)
        
        img_byte_arr = io.BytesIO()
        patch_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        print(f"Error creating patch image: {e}")
        return None


def create_tampered_version(orig_pdf_path: str, tamper_type: str) -> Optional[str]:
    """
    Apply forensically-relevant tampering to PDF with proper error handling.
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
                try:
                    doc.update_stream(xref, modified_content)
                except:
                    # Fallback if stream update fails
                    rect = fitz.Rect(-1000, -1000, -999, -999)
                    page.insert_textbox(rect, "HIDDEN_MARKER", fontsize=1, color=(1, 1, 1))
            else:
                rect = fitz.Rect(-1000, -1000, -999, -999)
                page.insert_textbox(rect, "HIDDEN_MARKER", fontsize=1, color=(1, 1, 1))

        elif tamper_type == "zero_width_space":
            rect = fitz.Rect(1, 1, 2, 2)
            page.insert_textbox(rect, "\u200b" * 15, fontsize=0.1, color=(1, 1, 1), render_mode=3)

        elif tamper_type == "javascript_injection":
            js_code = "app.alert('Document verification required');"
            js_bytes = js_code.encode("utf-8")
            try:
                doc.embfile_add("verification.js", js_bytes)
            except Exception as e:
                # Fallback: add as annotation
                page.add_text_annot(fitz.Point(5, 5), "JS_MARKER")

        elif tamper_type == "link_manipulation":
            links = page.get_links()
            if links:
                link = links[0]
                link['uri'] = "https://malicious-site.com/verify"
                page.update_link(link)
            else:
                rect = fitz.Rect(50, 50, 150, 70)
                page.insert_link({'kind': 2, 'from': rect, 'uri': 'https://phishing.com'})

        elif tamper_type == "improper_redaction":
            rl = page.search_for("the", quads=True)
            if rl:
                annot = page.add_redact_annot(rl[0], fill=(0,0,0))
                annot.update()
            else:
                rect = fitz.Rect(20, 20, 120, 40)
                annot = page.add_redact_annot(rect, fill=(0,0,0))
                annot.update()

        elif tamper_type == "image_splicing":
            img_list = page.get_images(full=True)
            if img_list:
                img_xref = img_list[0][0]
                img_rects = page.get_image_rects(img_xref)
                if img_rects:
                    rect = img_rects[0]
                    patch_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width*0.3, rect.y0 + rect.height*0.3)
                    patch_stream = _create_patch_image_stream()
                    if patch_stream:
                        page.insert_image(patch_rect, stream=patch_stream.read(), overlay=True)
            else:
                page.draw_line(fitz.Point(1, 1), fitz.Point(21, 1), color=(0.1, 0.1, 0.1))

        # === MEDIUM SEVERITY ATTACKS ===
        elif tamper_type == "font_substitution":
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
                              color=(0.9, 0.9, 0.9), rotate=0, overlay=False)

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
                "keywords": "Security Bypassed"
            })

        # === CRITICAL SEVERITY ATTACKS ===
        elif tamper_type == "content_stream_reorder":
            content_streams = _get_page_content_streams(page)
            if len(content_streams) > 1:
                try:
                    xref1, stream1 = content_streams[0]
                    xref2, stream2 = content_streams[1]
                    doc.update_stream(xref1, stream2)
                    doc.update_stream(xref2, stream1)
                except:
                    # Fallback if stream reordering fails
                    page.draw_rect(fitz.Rect(0, 0, 2, 2), color=(0.98, 0.98, 0.98))
            else:
                page.draw_rect(fitz.Rect(0, 0, 2, 2), color=(0.98, 0.98, 0.98))

        elif tamper_type == "page_deletion":
            if doc.page_count > 1:
                del_page_num = doc.page_count // 2
                doc.delete_page(del_page_num)
            else:
                page.draw_rect(fitz.Rect(1, 1, 3, 3), color=(0.98, 0.98, 0.98))

        elif tamper_type == "page_insertion":
            dummy_doc = fitz.open()
            dummy_page = dummy_doc.new_page()
            dummy_page.insert_text(fitz.Point(50, 72), "THIS IS A FAKE INSERTED PAGE")
            insert_at = doc.page_count // 2
            doc.insert_pdf(dummy_doc, from_page=0, to_page=0, start_at=insert_at)
            dummy_doc.close()

        else:
            print(f"Unknown tamper type: {tamper_type}")
            doc.close()
            return None
        
        # Save with incremental mode to preserve structure better
        doc.save(temp_path, garbage=4, deflate=True, clean=True)
        doc.close()
        
        # Verify the saved file is valid before moving
        try:
            test_doc = fitz.open(temp_path)
            test_doc.close()
        except Exception as e:
            print(f"  ⚠ Validation failed for {tamper_type}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
        
        if os.path.exists(tamp_pdf_path):
            os.remove(tamp_pdf_path)
        shutil.move(temp_path, tamp_pdf_path)
        
        return tamp_pdf_path

    except Exception as e:
        print(f"  ✗ Tampering error [{tamper_type}] on {base_name}: {e}")
        if doc:
            try:
                doc.close()
            except:
                pass
        for path in [temp_path, tamp_pdf_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        return None


# Severity mapping for analysis
SEVERITY_MAP = {
    "invisible_text": "HIGH",
    "zero_width_space": "HIGH",
    "javascript_injection": "HIGH",
    "link_manipulation": "HIGH",
    "improper_redaction": "HIGH",
    "image_splicing": "HIGH",
    
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
    
    "content_stream_reorder": "CRITICAL",
    "page_insertion": "CRITICAL",
    "page_deletion": "CRITICAL"
}


def generate_dataset_csv(num_samples_per_orig: int = SAMPLES_PER_ORIGINAL) -> None:
    """Generate dataset with identity pairs (label=0) and tampered pairs (label=1)."""
    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("ERROR: No PDF files in ORIGINAL_PDF_DIR")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files")
    print(f"Generating {num_samples_per_orig} tampered versions per PDF\n")
    
    data_log = []
    success = 0
    failed = 0
    severity_counts = {s: 0 for s in set(SEVERITY_MAP.values())}

    with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="pdf", dynamic_ncols=True, position=0) as pbar_pdfs:
        for orig_file in pdf_files:
            orig_path = os.path.join(ORIGINAL_PDF_DIR, orig_file)

                
            data_log.append([orig_path, orig_path, 0, "identity", "NONE"])
            pbar_pdfs.set_postfix_str(f"{orig_file}")
            
            
            with tqdm(total=num_samples_per_orig, desc="  Tampering", 
                      unit="op", leave=False, dynamic_ncols=True, position=1) as pbar_tamper:
                for _ in range(num_samples_per_orig):

                    t_type = random.choice(TAMPER_TYPES)
                    pbar_tamper.set_postfix_str(f"{t_type}")
                    
                    try:
                        tamp_path = create_tampered_version(orig_path, t_type)
                    except Exception as e:
                        # NEW: Catch the error here if it's high-level
                        tamp_path = None
                        print(f"\nCRITICAL MuPDF ERROR during tampering/save. Blacklisting {orig_file}. Error: {e}")
                    
                    if tamp_path:
                        severity = SEVERITY_MAP.get(t_type, "UNKNOWN")
                        data_log.append([orig_path, tamp_path, 1, t_type, severity])
                        success += 1
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    else:
                        failed += 1
                    
                    pbar_tamper.update(1)
            
            pbar_pdfs.update(1)
    
    print("\nSaving dataset to CSV...")
    with open(DATASET_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original_Path', 'Paired_Path', 'Label', 'Tamper_Type', 'Severity'])
        writer.writerows(data_log)
    
    print(f"\nDATASET GENERATION COMPLETE")
    print(f"Total pairs: {len(data_log)}")
    print(f"Tampered pairs: {success}")
    print(f"Failed attempts: {failed}")
    print(f"\nSeverity Distribution:")
    for severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = severity_counts.get(severity, 0)
        print(f"  {severity}: {count}")
    print(f"\nDataset saved to: {DATASET_CSV_PATH}")
    print(f"Tampered PDFs in: {TAMPERED_PDF_DIR}\n")
    
    
    print("\nSaving dataset to CSV...")
    with open(DATASET_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original_Path', 'Paired_Path', 'Label', 'Tamper_Type', 'Severity'])
        writer.writerows(data_log)
    
    print(f"\nDATASET GENERATION COMPLETE")
    print(f"Total pairs: {len(data_log)}")
    print(f"  Identity pairs: {len(pdf_files)}")
    print(f"  Tampered pairs: {success}")
    print(f"  Failed attempts: {failed}")
    print(f"\nSeverity Distribution:")
    for severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = severity_counts.get(severity, 0)
        print(f"  {severity}: {count}")
    print(f"\nDataset saved to: {DATASET_CSV_PATH}")
    print(f"Tampered PDFs in: {TAMPERED_PDF_DIR}\n")


if __name__ == "__main__":
    generate_dataset_csv()