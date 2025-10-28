import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from config import ORIGINAL_PDF_DIR, ARXIV_CATEGORY_URL, MAX_PDFS_TO_DOWNLOAD


def get_pdf_links(page_url):
    """Finds all PDF download links on the arXiv list page."""
    try:
        response = requests.get(page_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch URL: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all 'a' tags with the title 'Download PDF'
    pdf_links = []
    for link in soup.find_all('a', title='Download PDF'):
        href = link.get('href')
        if href and href.startswith('/pdf/'):
            # Construct the full URL
            full_url = f"https://arxiv.org{href}"
            pdf_links.append(full_url)
    
    return pdf_links

def download_pdf(pdf_url, save_path):
    """Downloads a single PDF with retries and progress."""
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdf_url}: {e}")
        return False

def main():
    print("Starting arXiv PDF Scraper...")
    print(f"Target directory: {ORIGINAL_PDF_DIR}")
    os.makedirs(ORIGINAL_PDF_DIR, exist_ok=True)
    
    print(f"Fetching PDF links from {ARXIV_CATEGORY_URL}...")
    pdf_links = get_pdf_links(ARXIV_CATEGORY_URL)
    
    if not pdf_links:
        print("No PDF links found. Exiting.")
        return
        
    print(f"Found {len(pdf_links)} total PDFs. Will attempt to download {MAX_PDFS_TO_DOWNLOAD}.")
    
    # Limit to the max number
    pdf_links = pdf_links[:MAX_PDFS_TO_DOWNLOAD]
    
    downloaded_count = 0
    for url in tqdm(pdf_links, desc="Downloading PDFs"):
        # Create a safe filename from the URL
        filename = url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename += ".pdf"
            
        save_path = os.path.join(ORIGINAL_PDF_DIR, filename)
        
        if os.path.exists(save_path):
            continue
            
        if download_pdf(url, save_path):
            downloaded_count += 1
            
        time.sleep(0.5) 

    print("\n" + "="*50)
    print("✓ Scraping Complete!")
    print(f"Successfully downloaded {downloaded_count} new PDFs.")
    print(f"Total PDFs in {ORIGINAL_PDF_DIR}: {len(os.listdir(ORIGINAL_PDF_DIR))}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()