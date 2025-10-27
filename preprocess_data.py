"""
Pre-processing script for the PDF Forensics Dataset.

This script runs ONCE after data_generator.py.
It reads the dataset.csv, extracts text and image modalities
from all unique PDFs, and saves them to disk.
This makes training dramatically faster by avoiding
on-the-fly PDF rendering.
"""

import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF
import io
import hashlib
from config import DATASET_CSV_PATH, IMAGE_SIZE

PREPROCESSED_DIR = "data/preprocessed"
IMG_DIR = os.path.join(PREPROCESSED_DIR, "images")
TXT_DIR = os.path.join(PREPROCESSED_DIR, "text")
NEW_CSV_PATH = os.path.join(PREPROCESSED_DIR, "dataset_preprocessed.csv")

# Create directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

def get_file_hash(path):
    """Creates a unique and safe filename hash from a path."""
    return hashlib.md5(path.encode()).hexdigest()

def extract_and_save(pdf_path: str, page_num: int = 0) -> tuple:
    """
    Extracts text and image from a PDF page and saves them to disk.
    Returns the paths to the saved text and image files.
    """
    try:
        # Create a unique hash for the (path, page) combination
        file_hash = f"{get_file_hash(pdf_path)}_p{page_num}"
        
        img_path = os.path.join(IMG_DIR, f"{file_hash}.jpg")
        txt_path = os.path.join(TXT_DIR, f"{file_hash}.txt")

        # If files already exist, skip processing
        if os.path.exists(img_path) and os.path.exists(txt_path):
            return txt_path, img_path

        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            return None, None
        
        # --- Improvement: Don't just use page 0 ---
        # Use the middle page for more representative content
        if page_num == -1: # Use -1 as a flag for "middle page"
             page_num = doc.page_count // 2
        
        # Ensure page_num is valid
        page_num = max(0, min(page_num, doc.page_count - 1))
        
        page = doc[page_num]

        # 1. Extract and save text
        text = page.get_text("text")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # 2. Extract and save image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better quality
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image.save(img_path, format="JPEG", quality=90)
        
        doc.close()
        return txt_path, img_path

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None, None

def main():
    print(f"Starting pre-processing. Outputting to {PREPROCESSED_DIR}")
    
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
    
    print(f"Found {len(all_unique_paths)} unique PDF documents to process.")

    # Process all unique PDFs and store their new paths
    # We use -1 to signal "extract middle page"
    processed_paths = {}
    for path in tqdm(all_unique_paths, desc="Processing unique PDFs"):
        txt_path, img_path = extract_and_save(path, page_num=-1)
        processed_paths[path] = (txt_path, img_path)

    # Create the new dataframe for the pre-processed dataset
    new_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating new dataset.csv"):
        orig_path = row['Original_Path']
        paired_path = row['Paired_Path']
        
        # Get the new pre-processed paths
        orig_txt, orig_img = processed_paths.get(orig_path, (None, None))
        paired_txt, paired_img = processed_paths.get(paired_path, (None, None))
        
        if orig_txt and paired_txt:
            new_data.append({
                'Original_Text_Path': orig_txt,
                'Original_Image_Path': orig_img,
                'Paired_Text_Path': paired_txt,
                'Paired_Image_Path': paired_img,
                'Label': row['Label'],
                'Tamper_Type': row['Tamper_Type'],
                'Severity': row['Severity']
            })

    # Save the new CSV
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(NEW_CSV_PATH, index=False)
    
    print("\n" + "="*50)
    print("✓ Pre-processing complete!")
    print(f"Saved {len(new_df)} samples to {NEW_CSV_PATH}")
    print("You can now run training (Option 3), which will be much faster.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()