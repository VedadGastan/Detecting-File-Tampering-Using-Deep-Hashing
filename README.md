# 🔎 PDF Forensics: Multimodal Deep Hashing System

This project implements a **Multimodal Deep Supervised Hashing** system designed to detect various forms of tampering in PDF documents. The system compares two PDF versions (or a document against a database) by generating a short, fixed-length binary **hash** from the document's visual and textual content. The similarity between two documents is then determined by calculating the **Hamming distance** between their hashes.

Optimized for reliable and fast training on CPU/Windows environments through a crucial data pre-processing step.

## 🚀 How It Works: The Core Idea

The system is built on a specialized deep neural network that combines two distinct modalities—**text** and **image**—to create a tamper-resistant hash.

### 1. Architecture (Model)

The system uses a fused multimodal architecture:
* **Text Modality:** A **DistilBERT** model processes the text content extracted from the PDF, capturing semantic meaning.
* **Image Modality:** An **EfficientNet-B0** model processes the rendered image of the PDF page, capturing visual layout and subtle graphic changes (like image splicing or improper redaction).
* **Fusion & Hashing:** The features from both branches are combined and passed through a final projection layer to produce a **64-bit binary hash**.

### 2. Forensic Data Generation

To train the model to be sensitive to tampering, the system automatically generates a dataset using **19 different forensically-relevant tampering techniques** (e.g., invisible text insertion, zero-width space attacks, image recompression, font substitution, etc.). The dataset consists of pairs of documents:
* **Identity Pair (Label 0):** Original PDF A compared against itself. (For training the hash to be stable and similar).
* **Tampered Pair (Label 1):** Original PDF A compared against Tampered PDF A'. (For training the hash to be sensitive and different).

### 3. Optimized Pipeline

Training deep learning models that rely on image data (like the EfficientNet branch) is computationally expensive, especially when rendering images from PDFs on-the-fly. The pipeline addresses this with a dedicated **Pre-processing Step** to speed up training:
1.  **Raw Data Generation (`data_generator.py`):** Creates the tampered PDFs and the initial `dataset.csv` (which contains paths to the raw PDF files).
2.  **Pre-processing (`preprocess_data.py`):** Renders every unique PDF in the raw dataset *once*, extracts the text and image, and saves them as dedicated `.txt` and `.png` files to disk.
3.  **Training (`train.py`):** The PyTorch `DataLoader` then reads the pre-saved `.txt` and `.png` files directly, eliminating the slow PDF rendering bottleneck during every training epoch.

## ⚙️ Setup and Installation

### Prerequisites

You need Python 3.8+ and the following system dependencies (for PyMuPDF):

* **Windows:** No special setup required.
* **Linux/macOS:** You may need development headers for PyMuPDF to install successfully.

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd pdf-forensics-hashing
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need a `requirements.txt` file listing all necessary packages like `torch`, `transformers`, `torchvision`, `pandas`, `tqdm`, `Pillow`, `PyMuPDF` (or `fitz`), `requests`, `beautifulsoup4`, `scikit-learn`, etc.)*

## 🏁 Usage: The Main Pipeline

The entire workflow is managed through the central script, `main.py`.

### 1. Initial Data Acquisition

First, you need to populate the source directory (`data/original_pdfs`) with documents.

```bash
python scrape_arxiv.py
````

This script downloads recent PDF papers from arXiv.org and saves them to the configured directory.

### 2\. Running the Pipeline

Execute the main script and select your desired option from the menu.

```bash
python main.py
```

You will be presented with the following options:

| Option | Description | Dependencies |
| :---: | :--- | :--- |
| **1** | **FULL RUN** | Requires: PDFs in `data/original_pdfs` |
| **2** | **Generate Raw Data Only** (Pairing & Tampering) | Requires: PDFs in `data/original_pdfs` |
| **3** | **Pre-process Data Only** | Requires: `data/dataset.csv` (from Option 2) |
| **4** | **Train Model Only** | Requires: Pre-processed data (from Option 3) |
| **5** | **Evaluate Model Only** | Requires: Trained model (`checkpoints/best_model.pth`) |

-----

### Recommended Workflow (Option 1)

For a first-time run, simply choose Option **1** for a full, automated process:

```
Pipeline Options:
 [1] FULL RUN (Generate → Preprocess → Train → Evaluate)
 [2] Generate Raw Data Only (Pairing & Tampering)
 [3] Pre-process Data Only (Requires dataset.csv)
 [4] Train Model Only (Requires pre-processed data)
 [5] Evaluate Model Only (Requires trained model)

Select Option [1-5] (default=1): 1
```

The script will proceed through the four main phases:

1.  **PHASE 1: Raw Data Generation:** Creates tampered PDFs and the initial dataset CSV.
2.  **PHASE 2: Data Pre-processing:** Extracts and saves text/image files for performance optimization.
3.  **PHASE 3: Model Training:** Trains the Multimodal Deep Hashing Model.
4.  **PHASE 4: Model Evaluation:** Computes standard retrieval metrics (MAP, P@K, R@K) and saves results.

## 🛠️ Configuration

All major parameters for the system are defined in `config.py`. You can adjust them to customize the process:

| Parameter | Location | Purpose |
| :--- | :--- | :--- |
| `DEVICE` | `config.py` | Set to `'cuda'` or `'cpu'`. |
| `HASH_BIT_LENGTH` | `config.py` | The length of the binary hash (e.g., `64`). |
| `SAMPLES_PER_ORIGINAL` | `config.py` | Number of tampered pairs generated for each original PDF (default `5`). |
| `NUM_EPOCHS` | `config.py` | Total training epochs. |
| `TAMPER_TYPES` | `config.py` | List of 19 tampering techniques used. |