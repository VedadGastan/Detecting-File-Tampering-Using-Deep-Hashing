import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from tqdm import tqdm

SAVE_DIRS = ["train_data", "test_data"]
MAX_PAPERS = 500

TRAIN_TOPICS = ["machine learning", "neural networks", "deep learning", "computer vision"]
TEST_TOPICS = ["cryptography", "cybersecurity", "quantum computing", "algorithms"]

def fetch_papers_for_dataset(target_dir, topics, papers_needed):
    count = 0
    failed = 0
    
    pbar = tqdm(total=papers_needed, desc=f"Downloading to {target_dir}", unit="papers")
    
    for topic in topics:
        if count >= papers_needed:
            break
            
        remaining = papers_needed - count
        url = f'http://export.arxiv.org/api/query?search_query=all:{topic.replace(" ", "+")}&start=0&max_results={remaining}&sortBy=submittedDate&sortOrder=descending'
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            data = response.read().decode('utf-8')
            
            root = ET.fromstring(data)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            for entry in entries:
                if count >= papers_needed:
                    break
                
                pdf_link = None
                for link in entry.findall('atom:link', namespace):
                    href = link.attrib.get('href', '')
                    link_type = link.attrib.get('type', '')
                    
                    if link_type == 'application/pdf' or href.endswith('.pdf'):
                        pdf_link = href
                        break
                
                if not pdf_link:
                    continue
                
                title_elem = entry.find('atom:title', namespace)
                if title_elem is None or not title_elem.text:
                    continue
                
                title = title_elem.text.strip().replace('\n', ' ').replace('\r', ' ')
                safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-')]).strip()[:50]
                paper_id = pdf_link.split('/')[-1].replace('.pdf', '')
                
                filename = f"{target_dir}/{paper_id}_{safe_title.replace(' ', '_')}.pdf"
                
                if not os.path.exists(filename):
                    try:
                        urllib.request.urlretrieve(pdf_link, filename)
                        
                        if os.path.getsize(filename) < 1000:
                            os.remove(filename)
                            failed += 1
                        else:
                            count += 1
                            pbar.update(1)
                            
                        time.sleep(3)
                            
                    except:
                        failed += 1
                        if os.path.exists(filename):
                            os.remove(filename)
                else:
                    count += 1
                    pbar.update(1)
                    
        except:
            pass
            
        if count < papers_needed:
            time.sleep(2)
    
    pbar.close()
    return count, failed

def fetch_arxiv_pdfs():
    for d in SAVE_DIRS:
        if not os.path.exists(d):
            os.makedirs(d)
    
    papers_per_dir = MAX_PAPERS // 2
    
    print(f"Fetching {papers_per_dir} training papers (ML/AI topics)...")
    train_count, train_failed = fetch_papers_for_dataset(SAVE_DIRS[0], TRAIN_TOPICS, papers_per_dir)
    
    print(f"\nFetching {papers_per_dir} testing papers (Security/Crypto topics)...")
    test_count, test_failed = fetch_papers_for_dataset(SAVE_DIRS[1], TEST_TOPICS, papers_per_dir)
    
    print(f"\nDownload complete:")
    print(f"  Training: {train_count} papers in {SAVE_DIRS[0]}/")
    print(f"  Testing:  {test_count} papers in {SAVE_DIRS[1]}/")
    print(f"  Failed:   {train_failed + test_failed} downloads")

if __name__ == "__main__":
    fetch_arxiv_pdfs()