"""
Enhanced Preprocessing with Structural Feature Extraction
Now saves: Text + Image + PDF Structure for each document
"""

import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import fitz
import io
import hashlib
import numpy as np
from config import DATASET_CSV_PATH, IMAGE_SIZE
from pdf_structural_features import extract_pdf_structural_features

PREPROCESSED_DIR = "data/preprocessed"
IMG_DIR = os.path.join(PREPROCESSED_DIR, "images")
TXT_DIR = os.path.join(PREPROCESSED_DIR, "text")
STRUCT_DIR = os.path.join(PREPROCESSED_DIR, "structural")  # NEW
NEW_CSV_PATH = os.path.join(PREPROCESSED_DIR, "dataset_preprocessed.csv")

# Create directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(STRUCT_DIR, exist_ok=True)


def get_file_hash(path):
    """Creates a unique and safe filename hash from a path."""
    return hashlib.md5(path.encode()).hexdigest()


def extract_and_save(pdf_path: str, page_num: int = 0) -> tuple:
    """
    Extracts text, image, AND structural features from a PDF page.
    Returns paths to all three saved files.
    """
    try:
        file_hash = f"{get_file_hash(pdf_path)}_p{page_num}"
        
        img_path = os.path.join(IMG_DIR, f"{file_hash}.jpg")
        txt_path = os.path.join(TXT_DIR, f"{file_hash}.txt")
        struct_path = os.path.join(STRUCT_DIR, f"{file_hash}.npy")  # NEW

        # If all files already exist, skip processing
        if os.path.exists(img_path) and os.path.exists(txt_path) and os.path.exists(struct_path):
            return txt_path, img_path, struct_path

        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            return None, None, None
        
        # Handle middle page selection
        if page_num == -1:
            page_num = doc.page_count // 2
        
        page_num = max(0, min(page_num, doc.page_count - 1))
        page = doc[page_num]

        # 1. Extract and save text
        text = page.get_text("text")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # 2. Extract and save image
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Reduced from 2x to 1.5x for speed
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image.save(img_path, format="JPEG", quality=85)  # Reduced quality for speed
        
        doc.close()
        
        # 3. NEW: Extract and save structural features
        structural_features = extract_pdf_structural_features(pdf_path, page_num)
        np.save(struct_path, structural_features)

        return txt_path, img_path, struct_path

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None, None, None


def main():
    print(f"\n{'='*60}")
    print("ENHANCED PRE-PROCESSING")
    print("Extracting: Text + Image + Structural Features")
    print(f"Output: {PREPROCESSED_DIR}")
    print(f"{'='*60}\n")
    
    # Load the original dataset.csv
    try:
        df = pd.read_csv(DATASET_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {DATASET_CSV_PATH} not found.")
        print("Please run 'python main.py' with option [2] (Generate Data) first.")
        return

    # Get all unique PDF paths
    paths1 = df['Original_Path'].unique()
    paths2 = df['Paired_Path'].unique()
    all_unique_paths = pd.Series(list(set(paths1) | set(paths2)))
    
    print(f"Found {len(all_unique_paths)} unique PDF documents to process.\n")

    # Process all unique PDFs and store their new paths
    processed_paths = {}
    failed_count = 0
    
    for path in tqdm(all_unique_paths, desc="Processing PDFs"):
        txt_path, img_path, struct_path = extract_and_save(path, page_num=-1)
        if txt_path and img_path and struct_path:
            processed_paths[path] = (txt_path, img_path, struct_path)
        else:
            failed_count += 1

    print(f"\nProcessing complete. Failed: {failed_count}/{len(all_unique_paths)}")

    # Create the new dataframe
    new_data = []
    skipped = 0
    
    print("\nCreating updated dataset CSV...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        orig_path = row['Original_Path']
        paired_path = row['Paired_Path']
        
        # Get the pre-processed paths
        orig_data = processed_paths.get(orig_path)
        paired_data = processed_paths.get(paired_path)
        
        if orig_data and paired_data:
            orig_txt, orig_img, orig_struct = orig_data
            paired_txt, paired_img, paired_struct = paired_data
            
            new_data.append({
                'Original_Text_Path': orig_txt,
                'Original_Image_Path': orig_img,
                'Original_Structural_Path': orig_struct,  # NEW
                'Paired_Text_Path': paired_txt,
                'Paired_Image_Path': paired_img,
                'Paired_Structural_Path': paired_struct,  # NEW
                'Label': row['Label'],
                'Tamper_Type': row['Tamper_Type'],
                'Severity': row['Severity']
            })
        else:
            skipped += 1

    # Save the new CSV
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(NEW_CSV_PATH, index=False)
    
    print(f"\n{'='*60}")
    print("✓ PRE-PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Processed samples: {len(new_df)}")
    print(f"Skipped (failed): {skipped}")
    print(f"Output CSV: {NEW_CSV_PATH}")
    print(f"\nFiles saved to:")
    print(f"  - Text: {TXT_DIR}")
    print(f"  - Images: {IMG_DIR}")
    print(f"  - Structural: {STRUCT_DIR}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()