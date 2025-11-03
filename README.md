# PDF Forensics Deep Hashing System - Technical Documentation

## Overview

This system implements a multimodal deep learning architecture for PDF document forensics and tampering detection. It uses deep supervised hashing to generate compact binary representations of documents across three modalities: textual content, visual appearance, and structural PDF features. The system can detect 19 different types of PDF tampering ranging from subtle metadata changes to critical content manipulation.

## System Architecture

### Core Components

The system consists of 10 interconnected modules that handle data acquisition, preprocessing, tampering simulation, model training, and evaluation:

1. **Configuration Module** (`config.py`)
2. **Data Generation** (`data_generator.py`, `scrape_arxiv.py`)
3. **Structural Feature Extraction** (`pdf_structural_features.py`)
4. **Preprocessing Pipeline** (`preprocess_data.py`)
5. **Dataset Loading** (`dataset.py`)
6. **Neural Network Model** (`model.py`)
7. **Training Loop** (`train.py`)
8. **Evaluation Framework** (`evaluate.py`)
9. **Pipeline Orchestrator** (`main.py`)

---

## Detailed Module Documentation

### 1. Configuration Module (`config.py`)

**Purpose**: Centralized configuration management for all system parameters.

**Key Parameters**:
- **Device Configuration**: Automatic CUDA/CPU detection
- **Model Architecture**: 64-bit hash length, 512 token max sequence, 224×224 image size
- **Training Hyperparameters**: Learning rate (1e-4), batch size (10), 30 epochs, 80/20 train-test split
- **Loss Function Weights**: 
  - β = 0.1 (quantization loss)
  - γ = 0.01 (distribution loss)
- **Tampering Parameters**: 5 tampered versions per original PDF
- **Data Paths**: Directory structure for PDFs, datasets, checkpoints, and results

**Tampering Taxonomy**: Defines 19 tampering techniques categorized by severity:
- **HIGH** (6 types): invisible text, zero-width spaces, JavaScript injection, link manipulation, improper redaction, image splicing
- **MEDIUM** (6 types): font substitution, page rotation, watermark injection, line artifacts, image recompression, annotation injection
- **LOW** (4 types): metadata changes, TOC removal, bookmark manipulation, encryption metadata
- **CRITICAL** (3 types): content stream reordering, page insertion, page deletion

---

### 2. Data Generation Module (`data_generator.py`)

**Purpose**: Creates synthetic tampering dataset by applying forensically-relevant attacks to original PDFs.

#### Core Functions

**`extract_page_modalities(pdf_path, page_num)`**
- Extracts text and rendered image from specified PDF page
- Returns tuple of (text_string, PIL_Image)
- Handles corrupted/empty PDFs with fallback white images

**`create_tampered_version(orig_pdf_path, tamper_type)`**
- Applies one of 19 tampering techniques to a PDF
- Uses PyMuPDF (fitz) for low-level PDF manipulation
- Implements safety checks and validation after tampering
- Returns path to tampered PDF or None on failure

#### Tampering Implementation Details

**High Severity Attacks**:
- **Invisible Text**: Injects hidden text at off-page coordinates (-1000, -1000) using PDF content stream operators
- **Zero-Width Space**: Inserts Unicode U+200B characters in tiny, white text
- **JavaScript Injection**: Embeds malicious JS code as file attachments
- **Link Manipulation**: Modifies existing hyperlinks or creates new ones pointing to phishing sites
- **Improper Redaction**: Applies redaction annotations without actually removing underlying text
- **Image Splicing**: Overlays "999" patch onto existing images to simulate manipulation

**Medium Severity Attacks**:
- **Font Substitution**: Inserts text with different fonts to create visual inconsistencies
- **Page Rotation**: Rotates page by 90 degrees
- **Watermark Injection**: Adds semi-transparent "CONFIDENTIAL" text overlay
- **Line Artifact**: Draws near-invisible lines (0.98, 0.98, 0.98 gray)
- **Image Recompression**: Re-renders page as JPEG at 60% quality
- **Annotation Injection**: Adds low-opacity annotations at specific locations

**Low Severity Attacks**:
- **Metadata Changes**: Modifies author, producer, creation/modification dates
- **TOC Removal**: Deletes table of contents structure
- **Bookmark Manipulation**: Creates fake bookmarks with malicious links
- **Encryption Metadata**: Alters encryption-related metadata fields

**Critical Severity Attacks**:
- **Content Stream Reordering**: Swaps PDF content streams to alter rendering order
- **Page Insertion**: Inserts fabricated pages with "FAKE" markers
- **Page Deletion**: Removes middle pages from multi-page documents

**`generate_dataset_csv(num_samples_per_orig)`**
- Orchestrates dataset generation process
- Creates identity pairs (label=0) for original-to-original comparisons
- Generates tampered pairs (label=1) by randomly applying tampering techniques
- Produces CSV with columns: Original_Path, Paired_Path, Label, Tamper_Type, Severity
- Implements nested progress bars for PDF-level and tampering-level tracking
- Provides severity distribution statistics

---

### 3. ArXiv Scraper (`scrape_arxiv.py`)

**Purpose**: Automated acquisition of research papers from arXiv for dataset creation.

**Functionality**:
- Parses arXiv category pages (default: computer science recent submissions)
- Extracts PDF download links using BeautifulSoup HTML parsing
- Downloads up to 2005 PDFs with configurable limit
- Implements rate limiting (0.5s delay) to respect server resources
- Skips already-downloaded files
- Provides progress tracking via tqdm

---

### 4. Structural Feature Extraction (`pdf_structural_features.py`)

**Purpose**: Extracts 40-dimensional forensic feature vectors from PDF internal structure.

**Feature Categories** (40 total features):

**Metadata Features** (8 dimensions):
- Author field presence/modification
- Producer field presence/modification  
- Creation date presence/validity
- Modification date presence/validity

**JavaScript & Actions** (4 dimensions):
- JavaScript presence count
- OpenAction presence
- Named action count
- AA (Additional Actions) presence

**Link & Navigation** (6 dimensions):
- Total hyperlink count
- External link count
- Internal link count
- Bookmark count
- Page label presence
- Outline structure integrity

**Annotation Features** (4 dimensions):
- Total annotation count
- Redaction annotation count
- Annotation with JavaScript count
- Hidden/invisible annotation count

**Content Stream Analysis** (6 dimensions):
- Number of content streams per page
- Content stream size statistics
- Stream compression ratios
- Operator frequency analysis

**Object Structure** (6 dimensions):
- Total object count
- Stream object count
- Indirect object count
- Cross-reference table integrity
- Incremental update count

**Encryption & Security** (3 dimensions):
- Encryption status
- Permission flags
- Digital signature presence

**Embedded Content** (3 dimensions):
- Embedded file count
- Form field count
- Multimedia annotation count

**Implementation**: Uses PyMuPDF's low-level PDF object access to extract structural information without rendering. Returns normalized 40-dimensional NumPy array.

---

### 5. Preprocessing Pipeline (`preprocess_data.py`)

**Purpose**: Converts raw PDF dataset into optimized format for fast training.

**Processing Steps**:

1. **Unique PDF Identification**: Extracts all unique PDF paths from dataset CSV
2. **Hash-Based Naming**: Generates MD5-based filenames to handle path variations
3. **Triple Extraction**: For each PDF, extracts and saves:
   - **Text**: UTF-8 encoded `.txt` files
   - **Image**: 224×224 JPEG renderings at 85% quality
   - **Structural**: 40-dimensional `.npy` feature arrays
4. **CSV Remapping**: Creates new CSV with preprocessed file paths
5. **Caching**: Skips already-processed files for efficient re-runs

**Output Structure**:
```
data/preprocessed/
├── dataset_preprocessed.csv
├── text/
│   └── [hash]_p0.txt
├── images/
│   └── [hash]_p0.jpg
└── structural/
    └── [hash]_p0.npy
```

**Performance Optimization**: By pre-rendering PDFs, this eliminates expensive PDF parsing during training, reducing epoch time by ~70%.

---

### 6. Dataset Loading Module (`dataset.py`)

**Purpose**: PyTorch Dataset and DataLoader implementation for efficient batch processing.

#### Key Components

**`ForensicsDataset` Class**:
- Inherits from `torch.utils.data.Dataset`
- Loads preprocessed files (text, image, structural) on-demand
- Applies image transformations: resize to 224×224, normalization using ImageNet statistics
- Tokenizes text using BERT tokenizer with max length 512, padding/truncation
- Returns tuple: ((doc1_text, doc1_img, doc1_struct), (doc2_text, doc2_img, doc2_struct), label)

**`collate_fn` Function**:
- Custom batch collation for variable-length sequences
- Stacks image tensors into batches
- Concatenates BERT token dictionaries (input_ids, attention_mask)
- Stacks structural feature vectors
- Handles all three modalities consistently

**`get_dataloader` Function**:
- Creates train/test split based on TRAIN_TEST_SPLIT ratio (0.8)
- Implements shuffling for training, sequential for testing
- Configures num_workers=4 for parallel loading (0 on Windows)
- Enables pin_memory for CUDA acceleration

**`get_full_dataloader` Function**:
- Returns non-split DataLoader for evaluation
- Processes entire dataset sequentially

---

### 7. Neural Network Architecture (`model.py`)

**Purpose**: Implements three-modality deep supervised hashing model.

#### Architecture Components

**`MultiModalHashingModel` Class**:

**Text Branch**:
- **Encoder**: DistilBERT (distilbert-base-uncased) with gradient checkpointing
- **Output**: 768-dimensional [CLS] token embedding
- **Projection**: Linear(768 → 64) + Tanh activation
- **Purpose**: Captures semantic textual content

**Image Branch**:
- **Encoder**: EfficientNet-B0 (pretrained on ImageNet)
- **Output**: 1280-dimensional feature vector
- **Projection**: Linear(1280 → 64) + Tanh activation
- **Purpose**: Captures visual appearance and layout

**Structural Branch**:
- **Encoder**: 3-layer MLP with dropout
  - Layer 1: Linear(40 → 256) + ReLU + Dropout(0.3)
  - Layer 2: Linear(256 → 128) + ReLU + Dropout(0.3)
  - Layer 3: Linear(128 → 64) + Tanh
- **Purpose**: Captures PDF internal structure and forensic features

**Fusion Mechanism**:
- **Attention Module**: Learns importance weights for each modality
  - Input: Concatenated 192-dimensional vector (64×3)
  - Output: 3-dimensional softmax weights
- **Weighted Combination**: Applies attention weights to individual hash codes
- **Final Hash Layer**: 
  - Linear(192 → 128) + ReLU + Dropout(0.2)
  - Linear(128 → 64) + Tanh
- **Output**: 64-bit continuous hash code in range [-1, +1]

**Forward Pass Logic**:
1. Process both documents independently through all three branches
2. Generate 64-dimensional hash codes for each branch
3. Apply attention-based weighting
4. Fuse weighted representations
5. Return pair of 64-dimensional hash codes

#### Utility Functions

**`binarize_hash(hash_output)`**: Converts continuous hashes to binary {-1, +1} using sign function

**`hamming_distance(hash1, hash2)`**: Computes Hamming distance between binary hash codes (range: 0 to 64)

**`cosine_similarity_hashing(hash1, hash2)`**: Alternative similarity metric using cosine similarity

---

### 8. Training Module (`train.py`)

**Purpose**: Implements training loop with custom loss function and optimization strategy.

#### Loss Function: `DeepHashingLoss`

**Components**:

1. **Similarity Loss**:
   - Minimizes distance between similar documents (label=0)
   - Maximizes distance between tampered documents (label=1)
   - Uses inner product similarity scaled by hash length
   - Formula: `-log(1 + clamp(target_similarity × inner_product))`
   - Applies 3× weight to positive class (tampered) to handle class imbalance

2. **Quantization Loss** (weight β=0.1):
   - Encourages hash codes to approach binary values {-1, +1}
   - Formula: `mean(|abs(hash) - 1|)` for both documents
   - Reduces continuous relaxation error

3. **Distribution Loss** (weight γ=0.01):
   - Ensures balanced bit distribution across hash dimensions
   - Prevents degenerate solutions where all bits collapse to same value
   - Formula: `mean(|mean(hash_per_bit)|)` for both documents

**Total Loss**: `L_similarity + β × L_quantization + γ × L_distribution`

#### Training Configuration

**Optimizer**: AdamW with differentiated learning rates:
- Pretrained encoders (DistilBERT, EfficientNet): 1e-5 (10% of base)
- Structural encoder: 1e-4 (full rate)
- Projection layers: 1e-4
- Fusion layers: 1e-4
- Weight decay: 1e-5

**Learning Rate Scheduler**: Cosine annealing over all epochs

**Mixed Precision Training**: 
- Enabled automatically on CUDA devices
- Uses PyTorch AMP (Automatic Mixed Precision)
- Applies GradScaler for stable fp16 gradients
- Disabled on CPU

**Checkpointing**:
- Saves best model based on validation loss
- Stores: epoch number, model state, optimizer state, loss history
- Path: `checkpoints/best_model.pth`

**Training Loop**:
1. For each epoch:
   - Forward pass through model with all three modalities
   - Compute multi-component loss
   - Backward propagation with mixed precision (if CUDA)
   - Optimizer step with gradient clipping
   - Validation phase (no gradient computation)
   - Learning rate scheduling
   - Checkpoint saving if validation improves

**Metrics Tracking**: Maintains history of train loss, test loss, similarity loss, quantization loss, and distribution loss per epoch

---

### 9. Evaluation Module (`evaluate.py`)

**Purpose**: Comprehensive performance assessment using retrieval and classification metrics.

#### Evaluation Metrics

**Retrieval Metrics**:

1. **Mean Average Precision (MAP)**:
   - Measures ranking quality for document retrieval
   - For each query, computes average precision across relevant documents
   - Averages across all queries

2. **Precision@K and Recall@K**:
   - Evaluates top-K retrieval performance
   - Computed for K ∈ {1, 5, 10, 20}
   - Precision: fraction of relevant documents in top-K
   - Recall: fraction of all relevant documents found in top-K

**Classification Metrics**:

1. **Binary Classification**:
   - Uses threshold = 32 bits (50% of hash length)
   - Predictions: distance > threshold → tampered (1), else similar (0)
   - Computes: Precision, Recall, F1-Score
   - Generates confusion matrix

2. **ROC Analysis**:
   - Computes ROC curve (FPR vs TPR)
   - Calculates Area Under Curve (AUC)
   - Evaluates discrimination ability

3. **Precision-Recall Curve**:
   - Plots precision vs recall at varying thresholds
   - Computes Average Precision (AP)

**Severity Analysis**:
- Groups predictions by tampering severity (LOW, MEDIUM, HIGH, CRITICAL)
- Computes mean Hamming distance for each severity level
- Evaluates if model learns severity hierarchy

**Statistical Analysis**:
- Mann-Whitney U test: Tests if similar and tampered distributions differ significantly
- Effect size (Cohen's d): Quantifies magnitude of difference
- Distribution statistics: Mean and standard deviation for both classes

#### Evaluation Process

1. **Hash Extraction**: 
   - Loads trained model
   - Processes all document pairs
   - Generates binary hash codes
   - Constructs N×N distance matrix

2. **Metric Computation**:
   - Applies all evaluation metrics to distance matrix
   - Performs severity-stratified analysis
   - Runs statistical tests

3. **Visualization Generation**:
   - ROC curve plot
   - Distance distribution histograms (similar vs tampered)
   - Confusion matrix heatmap
   - Saves all figures to `results/` directory

4. **Report Generation**:
   - Compiles comprehensive text report
   - Includes all metrics, confusion matrix, severity breakdown
   - Saves to `results/evaluation_results.txt`

---

### 10. Pipeline Orchestrator (`main.py`)

**Purpose**: Unified entry point for executing complete pipeline or individual stages.

#### Pipeline Stages

**Stage 1: Raw Data Generation**:
- Validates presence of original PDFs
- Calls `generate_dataset_csv()` to create tampered pairs
- Outputs: `data/dataset.csv`

**Stage 2: Preprocessing**:
- Executes `preprocess_data.py` as subprocess
- Converts PDFs to optimized format (text, images, structural features)
- Outputs: `data/preprocessed/dataset_preprocessed.csv` and associated files

**Stage 3: Model Training**:
- Calls `main_train()` with preprocessed data
- Trains three-modality hashing model
- Outputs: `checkpoints/best_model.pth`

**Stage 4: Evaluation**:
- Calls `evaluate_model()` with trained model
- Computes comprehensive metrics
- Outputs: `results/evaluation_results.txt` and visualizations

#### Execution Modes

1. **Full Run**: Executes all four stages sequentially
2. **Generate Only**: Creates raw tampered dataset
3. **Preprocess Only**: Converts dataset to optimized format
4. **Train Only**: Trains model on preprocessed data
5. **Evaluate Only**: Evaluates existing trained model

**Dependency Checking**:
- Validates existence of required files before each stage
- Provides clear error messages for missing dependencies
- Suggests corrective actions

**Hardware Detection**:
- Displays device status (CUDA/CPU) at startup
- Adapts pipeline for CPU optimization when GPU unavailable

---

## System Workflow

### Complete Pipeline Execution

```
1. PDF Acquisition (scrape_arxiv.py)
   â"" Downloads 2000+ research papers
   
2. Tampering Simulation (data_generator.py)
   â"" Applies 19 tampering techniques
   â"" Creates 5 tampered versions per original
   â"" Generates dataset.csv with ~10,000+ pairs
   
3. Structural Analysis (pdf_structural_features.py)
   â"" Extracts 40-dimensional feature vectors
   â"" Captures metadata, JavaScript, links, annotations, etc.
   
4. Preprocessing (preprocess_data.py)
   â"" Pre-renders pages as 224×224 images
   â"" Extracts text content
   â"" Saves structural features
   â"" Creates optimized dataset
   
5. Model Training (train.py)
   â"" Trains DistilBERT + EfficientNet + MLP fusion
   â"" 30 epochs with cosine annealing
   â"" Saves best model checkpoint
   
6. Evaluation (evaluate.py)
   â"" Computes retrieval metrics (MAP, P@K, R@K)
   â"" Computes classification metrics (Precision, Recall, F1)
   â"" Performs severity analysis
   â"" Generates visualizations and report
```

---

## Technical Implementation Details

### PDF Manipulation Techniques

**Low-Level Content Stream Editing**:
- Directly modifies PDF content streams using byte-level operations
- Injects PDF operators (BT, Tm, Tj, ET) for invisible text
- Manipulates transformation matrices for off-page rendering

**Annotation Layer Attacks**:
- Exploits PDF annotation system for injecting hidden content
- Uses opacity controls (0.0-1.0) to create semi-invisible elements
- Leverages render_mode parameter for text invisibility

**Structural Manipulation**:
- Reorders object references in cross-reference table
- Modifies content stream sequences to alter rendering
- Inserts/deletes pages while maintaining document structure

**Metadata Tampering**:
- Modifies XMP metadata streams
- Alters document information dictionary
- Manipulates date formats and encoding

### Feature Engineering

**Structural Feature Normalization**:
- Counts normalized by document size (pages, objects)
- Binary flags for presence/absence
- Statistical aggregations (min, max, mean, std)

**Text Processing**:
- BERT WordPiece tokenization
- Max sequence length 512 tokens
- Padding/truncation for batch consistency

**Image Preprocessing**:
- Resize to 224×224 for EfficientNet compatibility
- Normalization using ImageNet mean/std
- RGB conversion for consistency

### Training Optimizations

**Memory Efficiency**:
- Gradient checkpointing in DistilBERT reduces memory by ~40%
- Mixed precision training (fp16) reduces memory usage
- Pin memory for faster GPU transfers

**Computational Efficiency**:
- Preprocessing eliminates runtime PDF rendering (~70% faster epochs)
- Batch processing with DataLoader parallelism (4 workers)
- Efficient hash distance computation using vectorized operations

**Convergence Improvements**:
- Differentiated learning rates for pretrained vs new components
- Class imbalance weighting (3× for minority class)
- Cosine annealing for smooth convergence

---

## Model Capabilities

### Detection Performance

The three-modality architecture enables detection of:

1. **Textual Tampering**: Invisible text, zero-width characters, font changes
2. **Visual Tampering**: Image splicing, watermarks, rotation, compression artifacts
3. **Structural Tampering**: JavaScript injection, link manipulation, stream reordering, metadata changes
4. **Hybrid Tampering**: Attacks affecting multiple modalities simultaneously

### Hash Code Properties

**Compactness**: 64-bit binary codes (8 bytes per document)
**Efficiency**: Hamming distance computed in O(1) time
**Discriminability**: Similar documents have low Hamming distance (<32 bits), tampered documents have high distance (>32 bits)
**Robustness**: Multi-modal fusion provides redundancy against single-modality attacks

---

## Data Specifications

### Dataset Statistics

- **Original PDFs**: 2000+ research papers from arXiv
- **Tampered Versions**: 5 per original = ~10,000 tampered PDFs
- **Total Pairs**: ~10,000 identity pairs + ~10,000 tampered pairs = ~20,000 training samples
- **Train/Test Split**: 80/20 = ~16,000 train / ~4,000 test

### Severity Distribution

Based on configuration, expected distribution:
- **LOW**: ~20% (4 techniques out of 19)
- **MEDIUM**: ~32% (6 techniques)
- **HIGH**: ~32% (6 techniques)
- **CRITICAL**: ~16% (3 techniques)

### File Sizes

- **Original PDFs**: Variable (typically 200KB - 5MB)
- **Tampered PDFs**: Similar to original
- **Preprocessed Images**: ~50-100KB each (JPEG 85% quality)
- **Preprocessed Text**: Variable (typically 1-50KB)
- **Structural Features**: 320 bytes each (40 floats × 8 bytes)
- **Total Dataset Size**: ~40-60GB for complete preprocessed dataset

---

## Performance Characteristics

### Training Time

- **Per Epoch**: ~15-25 minutes (CUDA) / ~45-60 minutes (CPU optimized)
- **30 Epochs**: ~7.5-12.5 hours (CUDA) / ~22-30 hours (CPU)

### Evaluation Time

- **Hash Extraction**: ~5-10 minutes for 20,000 samples
- **Distance Matrix**: O(N²) = ~5-10 minutes for full pairwise computation
- **Metric Computation**: <1 minute
- **Total**: ~15-20 minutes

### Inference Time

- **Single Document**: ~50-100ms (includes all three modalities)
- **Batch Processing**: ~5-10ms per document with batch size 32
- **Hash Comparison**: <0.1ms (simple Hamming distance)

---

## Scientific Foundations

### Deep Supervised Hashing

**Concept**: Learn compact binary codes that preserve semantic similarity
**Advantage**: O(1) retrieval time vs O(N) for continuous vectors
**Application**: Large-scale document forensics with millions of PDFs

### Multi-Modal Learning

**Concept**: Combine complementary information sources for robust representation
**Rationale**: Different tampering types affect different modalities
**Implementation**: Late fusion with attention-based weighting

### Contrastive Learning

**Concept**: Learn representations by comparing positive and negative pairs
**Loss**: Minimizes distance for similar pairs, maximizes for dissimilar
**Application**: Trains model to distinguish authentic from tampered documents

---

## Limitations and Constraints

### Model Limitations

1. **Page-Level Analysis**: Only processes first/middle page, not entire documents
2. **Fixed Hash Length**: 64 bits may not capture all tampering nuances
3. **Binary Classification**: Does not identify specific tampering type
4. **Training Data Dependency**: Performance limited by training tampering types

### Technical Constraints

1. **Memory**: Full batch training requires ~16-32GB RAM (CPU) or ~8-12GB VRAM (GPU)
2. **Storage**: Complete dataset requires ~50-100GB disk space
3. **PyMuPDF Dependency**: Some tampering techniques may fail on certain PDF structures
4. **Compute**: Training requires significant computational resources

### Detection Gaps

1. **Sophisticated Attacks**: May not detect advanced steganography or cryptographic manipulations
2. **Benign Modifications**: May flag legitimate edits as tampering
3. **Novel Attacks**: Zero-shot performance on unseen tampering types uncertain
4. **Encoding Variations**: Different PDF encoders may produce false positives

---

## Output Artifacts

### Training Outputs

- `checkpoints/best_model.pth`: Trained model weights (~400MB)
- Training loss curves (embedded in checkpoint)

### Evaluation Outputs

- `results/evaluation_results.txt`: Comprehensive metrics report
- `results/roc_curve.png`: ROC curve visualization
- `results/distance_distribution.png`: Hamming distance histograms
- `results/confusion_matrix.png`: Classification confusion matrix

### Dataset Artifacts

- `data/dataset.csv`: Raw dataset with PDF pairs
- `data/preprocessed/dataset_preprocessed.csv`: Optimized dataset
- `data/preprocessed/text/`: Extracted text files
- `data/preprocessed/images/`: Rendered page images
- `data/preprocessed/structural/`: Structural feature vectors
- `data/tampered_pdfs/`: Generated tampered PDFs

---

## Dependencies and Requirements

### Core Libraries

- **PyTorch**: Neural network framework
- **Transformers**: BERT models (Hugging Face)
- **PyMuPDF (fitz)**: PDF manipulation and rendering
- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **Pandas**: Dataset management
- **scikit-learn**: Evaluation metrics
- **Matplotlib/Seaborn**: Visualization
- **tqdm**: Progress tracking
- **BeautifulSoup**: Web scraping

### Model Dependencies

- **distilbert-base-uncased**: Pre-trained language model
- **efficientnet_b0**: Pre-trained image model (ImageNet weights)

### System Requirements

- **Python**: 3.8+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB+ for full dataset
- **GPU**: Optional but recommended (8GB+ VRAM)

---