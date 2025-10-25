import os
import requests
import tarfile
import shutil
from tqdm import tqdm
from config import ORIGINAL_PDF_DIR

# --- Configuration for Download ---
# NOTE: Replace this with the actual URL to a bulk data manifest or tar file.
# For arXiv, this is typically a link to a large tar.gz file of a specific category/month.
# A small, hypothetical bulk file is used here as a placeholder for the actual large link.
BULK_DATA_URL = "http://example.com/data/arxiv_astro_ph_2024.tar.gz"
DOWNLOAD_FILENAME = "arxiv_bulk_data.tar.gz"
EXTRACT_DIR = "temp_pdf_extract"

def download_file(url, filename):
    """Robustly downloads a file with progress bar and handles stream."""
    print(f"Attempting to download bulk file from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print(f"✅ Download complete: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"🚨 ERROR: Download failed. Check URL or internet connection. ({e})")
        print("Using official bulk data access is required to respect policies.")
        return False


def extract_and_move_pdfs(archive_path, target_dir):
    """Extracts PDFs from the archive and moves them to the original data directory."""
    print(f"Starting extraction of {archive_path}...")
    
    # 1. Ensure temporary and target directories exist
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # 2. Extract the archive
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Safely extract all members to the temporary directory
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner) 
            
            safe_extract(tar, path=EXTRACT_DIR)
        print(f"Extraction to {EXTRACT_DIR} complete.")
    except tarfile.TarError as e:
        print(f"🚨 ERROR: Failed to extract archive. Is it a valid .tar.gz file? ({e})")
        return

    # 3. Move all .pdf files to the final data directory
    pdf_count = 0
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".pdf"):
                src_path = os.path.join(root, file)
                # Rename the file to ensure uniqueness (e.g., use an ID or a hash)
                dst_path = os.path.join(target_dir, f"{pdf_count}_{file}") 
                shutil.move(src_path, dst_path)
                pdf_count += 1
    
    print(f"✅ Successfully moved {pdf_count} PDF files to {target_dir}.")

    # 4. Cleanup
    shutil.rmtree(EXTRACT_DIR)
    os.remove(archive_path)
    print("Cleanup successful.")

def main_scraper_routine():
    """Main routine to acquire the PDF dataset."""
    
    # 1. Setup
    os.makedirs(ORIGINAL_PDF_DIR, exist_ok=True)
    
    # 2. Download the bulk archive
    if download_file(BULK_DATA_URL, DOWNLOAD_FILENAME):
        
        # 3. Extract and move PDFs
        extract_and_move_pdfs(DOWNLOAD_FILENAME, ORIGINAL_PDF_DIR)