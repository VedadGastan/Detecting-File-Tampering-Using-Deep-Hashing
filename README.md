# Multimodal Deep Hashing for PDF Forensics

**Bachelor Thesis Implementation**: Detecting PDF Document Tampering using Deep Learning

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tampering Detection Methods](#tampering-detection-methods)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Model Architecture Details](#model-architecture-details)
7. [Evaluation Metrics Explained](#evaluation-metrics-explained)
8. [File Structure](#file-structure)
9. [Configuration](#configuration)
10. [Results Interpretation](#results-interpretation)

---

## 🎯 Overview

This system detects tampering in PDF documents using **deep supervised hashing** with multimodal features (text + images). It generates compact binary hash codes that cluster similar documents together while separating tampered ones.

### Key Features

- **15 Forensically-Relevant Tampering Techniques**
- **Multimodal Deep Learning**: BERT (text) + ResNet (image)
- **Fast Training**: Mixed precision training with AMP
- **Scientific Evaluation**: ROC curves, statistical tests, retrieval metrics
- **Severity Classification**: LOW, MEDIUM, HIGH, CRITICAL tampering levels

### How It Works

```
PDF Document → Text + Image Extraction → BERT + ResNet → Hash Code (64 bits)
                                            ↓
                                    Similar PDFs = Similar Hashes
                                    Tampered PDFs = Different Hashes
```

---

## 🏗️ Architecture

### Model Components

```
┌─────────────────────────────────────────────────────┐
│                   INPUT: PDF Page                   │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼───┐        ┌────▼────┐
    │ TEXT  │        │  IMAGE  │
    └───┬───┘        └────┬────┘
        │                 │
    ┌───▼───────┐    ┌────▼─────────┐
    │   BERT    │    │  ResNet-18   │
    │ (768-dim) │    │  (512-dim)   │
    └───┬───────┘    └────┬─────────┘
        │                 │
    ┌───▼───────┐    ┌────▼─────────┐
    │ Project   │    │   Project    │
    │ to 64-bit │    │  to 64-bit   │
    └───┬───────┘    └────┬─────────┘
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Fusion Layer   │
        │    (64-bit)     │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Binary Hash    │
        │   {-1, +1}^64   │
        └─────────────────┘
```

### Text Encoder: BERT

**Model**: `bert-base-uncased` (110M parameters)
- **Input**: Tokenized text (max 512 tokens)
- **Output**: 768-dimensional embedding from [CLS] token
- **Captures**: Semantic text content, document structure

**Why BERT?**
- Pretrained on massive text corpora
- Understands context and semantics
- Robust to minor text variations

**Alternatives**:
- **RoBERTa**: Better performance, more training data
- **DistilBERT**: 40% smaller, 60% faster, 97% performance
- **ELECTRA**: More efficient pretraining
- **ALBERT**: Parameter-efficient, good for limited memory
- **DeBERTa**: State-of-the-art, larger models

### Image Encoder: ResNet-18

**Model**: ResNet-18 pretrained on ImageNet (11M parameters)
- **Input**: 224×224 RGB image (rendered PDF page)
- **Output**: 512-dimensional feature vector
- **Captures**: Visual layout, formatting, artifacts

**Why ResNet-18?**
- Lightweight but effective
- Pretrained on ImageNet (good feature extractor)
- Fast inference

**Alternatives**:
- **EfficientNet-B0**: Better accuracy/efficiency trade-off
- **Vision Transformer (ViT)**: Attention-based, state-of-the-art
- **ConvNeXT**: Modern CNN architecture
- **DenseNet**: Better feature reuse
- **MobileNetV3**: Very fast, mobile-friendly
- **Swin Transformer**: Hierarchical vision transformer

### Hash Projection

**Deep Supervised Hashing (DSH)**:
1. **Projection**: Linear layer + Tanh → outputs in [-1, 1]
2. **Binarization**: sign(x) → converts to {-1, +1}
3. **Loss Function**:
   - **Similarity Loss**: Similar docs → small Hamming distance
   - **Quantization Loss**: Forces outputs to ±1
   - **Distribution Loss**: Balances bit usage

---

## 🔍 Tampering Detection Methods

### All 15 Tampering Techniques

| # | Technique | Severity | Description | Detection Importance |
|---|-----------|----------|-------------|---------------------|
| 1 | **Invisible Text Injection** | HIGH | Hidden text off-page boundaries | Common in fraud, phishing |
| 2 | **Zero-Width Space Injection** | HIGH | Invisible Unicode characters | Text manipulation, OCR evasion |
| 3 | **JavaScript Injection** | HIGH | Malicious code embedding | Security threat, payload delivery |
| 4 | **Link Manipulation** | HIGH | URL/hyperlink tampering | Phishing, redirect attacks |
| 5 | **Font Substitution** | MEDIUM | Font replacement attacks | Visual spoofing |
| 6 | **Page Rotation** | MEDIUM | Orientation changes | Layout tampering |
| 7 | **Watermark Injection** | MEDIUM | Hidden/visible watermark addition | Authenticity claims |
| 8 | **Line Artifacts** | MEDIUM | Near-invisible visual marks | Subtle visual tampering |
| 9 | **Image Recompression** | MEDIUM | Quality degradation | Hides edits, reduces file size |
| 10 | **Annotation Injection** | MEDIUM | Hidden comments/notes | Metadata tampering |
| 11 | **Metadata Changes** | LOW | Author, date, producer modification | Basic tampering indicator |
| 12 | **TOC Removal** | LOW | Table of contents deletion | Structural change |
| 13 | **Bookmark Manipulation** | LOW | PDF outline changes | Navigation tampering |
| 14 | **Encryption Metadata** | LOW | Security settings changes | False security claims |
| 15 | **Content Stream Reordering** | CRITICAL | PDF object reordering | Deep structural tampering |

### Why These Matter

**High Severity**: Direct security threats, common in malicious documents
**Medium Severity**: Visual/structural tampering, harder to detect manually
**Low Severity**: Metadata tampering, easy to detect but still relevant
**Critical Severity**: Advanced attacks requiring PDF structure knowledge

---

## 📦 Installation

### Requirements

```bash
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd pdf-forensics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
PyMuPDF>=1.22.0
Pillow>=9.5.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 🚀 Quick Start

### 1. Prepare Data

Add your PDF files to `data/original_pdfs/`:

```bash
mkdir -p data/original_pdfs
cp /path/to/your/pdfs/*.pdf data/original_pdfs/
```

**Recommended**: 50-100 diverse PDF documents (academic papers, reports, etc.)

### 2. Run Full Pipeline

```bash
python main.py
```

**Options**:
- `[1]` Full pipeline (recommended first run)
- `[2]` Generate data only
- `[3]` Train model only
- `[4]` Evaluate model only

### 3. View Results

```
results/
├── evaluation_results.txt      # Numerical metrics
├── roc_curve.png               # ROC curve visualization
├── pr_curve.png                # Precision-Recall curve
├── distance_distribution.png   # Hamming distance histogram
├── confusion_matrix.png        # Classification matrix
└── training_curves.png         # Training history (if available)
```

---

## 🧠 Model Architecture Details

### Training Process

```python
# 1. Data Generation
for each PDF:
    Original vs Original → Label = 0 (similar)
    Original vs Tampered → Label = 1 (dissimilar)

# 2. Forward Pass
text_embed = BERT(text_tokens)           # → 768-dim
image_embed = ResNet(image_pixels)       # → 512-dim
text_hash = project(text_embed)          # → 64-dim
image_hash = project(image_embed)        # → 64-dim
final_hash = fusion([text_hash, image_hash])  # → 64-dim

# 3. Loss Computation
similarity_loss = log(1 + exp(-target * inner_product))
quantization_loss = (|hash| - 1)²
distribution_loss = mean(hash)²
total_loss = similarity_loss + β*quantization_loss + γ*distribution_loss

# 4. Binarization (inference)
binary_hash = sign(final_hash)  # {-1, +1}^64
```

### Hyperparameters

```python
HASH_BIT_LENGTH = 64        # Hash code size
LEARNING_RATE = 1e-4        # AdamW optimizer
BATCH_SIZE = 8              # Batch size
NUM_EPOCHS = 20             # Training epochs
BETA_QUANT = 0.1            # Quantization loss weight
GAMMA_DIST = 0.01           # Distribution loss weight
```

### Speed Optimizations

1. **Mixed Precision Training** (AMP): 2-3× faster on GPU
2. **Cosine Annealing Scheduler**: Better convergence
3. **AdamW Optimizer**: Decoupled weight decay
4. **Gradient Checkpointing** (optional): Reduced memory usage

---

## 📊 Evaluation Metrics Explained

### Retrieval Metrics

#### Mean Average Precision (MAP)

**What it measures**: How well the model ranks similar documents

**Calculation**:
```
For each query document:
    1. Rank all documents by Hamming distance
    2. Compute Precision@k for each relevant document position
    3. Average these precisions → AP
MAP = mean of all APs
```

**Interpretation**:
- MAP = 1.0: Perfect ranking
- MAP = 0.5: Random ranking
- MAP > 0.8: Excellent performance

#### Precision@K and Recall@K

**Precision@K**: % of top-K results that are relevant
```
Precision@K = (Relevant docs in top-K) / K
```

**Recall@K**: % of all relevant docs found in top-K
```
Recall@K = (Relevant docs in top-K) / (Total relevant docs)
```

**Trade-off**: Higher K → Higher Recall, Lower Precision

### Classification Metrics

#### Confusion Matrix

```
                Predicted
              Similar  Tampered
Actual Similar    TP       FN
       Tampered   FP       TN
```

- **True Positive (TP)**: Correctly identified similar docs
- **False Negative (FN)**: Similar docs misclassified as tampered
- **False Positive (FP)**: Tampered docs misclassified as similar
- **True Negative (TN)**: Correctly identified tampered docs

#### Precision, Recall, F1-Score

```
Precision = TP / (TP + FP)   # How many predictions are correct
Recall = TP / (TP + FN)      # How many actual similar docs found
F1 = 2 * (Precision * Recall) / (Precision + Recall)  # Harmonic mean
```

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic)**:
- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR)
- Shows trade-off at different thresholds

**AUC (Area Under Curve)**:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC > 0.9: Excellent model

### Precision-Recall Curve

**Better for imbalanced datasets** (more tampered than similar docs)
- Shows precision vs recall at different thresholds
- Average Precision = area under PR curve

### Statistical Tests

#### Mann-Whitney U Test

**What it tests**: Are similar and tampered distance distributions different?

**Null hypothesis**: Distributions are the same
**p-value < 0.05**: Reject null hypothesis (distributions ARE different)

#### Cohen's d (Effect Size)

**Measures magnitude of difference**:
```
d = (mean_similar - mean_tampered) / pooled_std
```

**Interpretation**:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect
- d > 1.0: Very large effect

### Hash Quality Metrics

#### Bit Balance

**Measures**: How evenly distributed are +1/-1 values across bits
```
bit_balance = mean(|bit_mean - 0.5|)
```
**Good hash**: bit_balance → 0 (each bit is 50/50)

#### Bit Variance

**Measures**: How much each bit varies across samples
```
bit_variance = mean(variance(bit_i))
```
**Good hash**: High variance (bits discriminate well)

---

## 📁 File Structure

```
pdf-forensics/
│
├── main.py                 # Main pipeline orchestrator
├── config.py               # Configuration parameters
├── model.py                # MultiModalHashingModel definition
├── train.py                # Training loop with AMP
├── evaluate.py             # Comprehensive evaluation suite
├── dataset.py              # PyTorch Dataset and DataLoader
├── data_generator.py       # PDF tampering and dataset creation
│
├── data/
│   ├── original_pdfs/      # Your original PDF files
│   ├── tampered_pdfs/      # Generated tampered versions
│   └── dataset.csv         # Paired dataset metadata
│
├── checkpoints/
│   └── best_model.pth      # Trained model weights
│
├── results/
│   ├── evaluation_results.txt
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── distance_distribution.png
│   └── confusion_matrix.png
│
└── README.md               # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model
HASH_BIT_LENGTH = 64          # Hash code size (32, 64, 128)
MAX_SEQ_LENGTH = 512          # BERT max tokens
IMAGE_SIZE = 224              # ResNet input size

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 8                # Reduce if out of memory
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.8

# Loss weights
BETA_QUANT = 0.1
GAMMA_DIST = 0.01

# Data generation
SAMPLES_PER_ORIGINAL = 5      # Tampered versions per PDF
TAMPER_TYPES = [...]          # Enable/disable tampering types

# Paths
ORIGINAL_PDF_DIR = "data/original_pdfs"
TAMPERED_PDF_DIR = "data/tampered_pdfs"
MODEL_SAVE_PATH = "checkpoints/best_model.pth"
RESULTS_DIR = "results"

# Evaluation
TOP_K_VALUES = [1, 5, 10, 20]  # K values for P@K and R@K
```

---

## 📈 Results Interpretation

### Good Model Indicators

✅ **MAP > 0.8**: Excellent retrieval performance
✅ **ROC AUC > 0.9**: Strong discrimination ability
✅ **F1-Score > 0.85**: Good balance of precision/recall
✅ **p-value < 0.001**: Statistically significant difference
✅ **Cohen's d > 1.0**: Large effect size
✅ **Bit balance < 0.1**: Well-distributed hash codes

### Common Issues

❌ **Low MAP (<0.6)**: Model not learning similarity well
- Solution: Train longer, adjust loss weights, check data quality

❌ **High FP rate**: Tampered docs misclassified as similar
- Solution: Lower threshold, add more training data

❌ **High FN rate**: Similar docs misclassified as tampered
- Solution: Increase threshold, improve model capacity

❌ **Poor bit balance (>0.2)**: Hash codes not utilizing all bits
- Solution: Increase GAMMA_DIST, check distribution loss

---

## 🔬 Advanced Usage

### Custom Tampering

Add your own tampering in `data_generator.py`:

```python
def create_tampered_version(orig_pdf_path, tamper_type):
    # ...
    elif tamper_type == "my_custom_attack":
        # Your tampering logic here
        page.custom_operation()
```

### Different Architectures

Replace encoders in `model.py`:

```python
# Use RoBERTa instead of BERT
self.text_encoder = AutoModel.from_pretrained('roberta-base')

# Use EfficientNet instead of ResNet
from efficientnet_pytorch import EfficientNet
self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
```

### Severity-Aware Loss

Modify loss function in `train.py`:

```python
# Weight loss by severity
severity_weights = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 1.5, 'CRITICAL': 2.0}
weighted_loss = loss * severity_weights[tamper_severity]
```

---

## 📚 References

### Papers

1. **Deep Supervised Hashing**: Lin et al. (CVPR 2015)
2. **BERT**: Devlin et al. (NAACL 2019)
3. **ResNet**: He et al. (CVPR 2016)

### Libraries

- **PyTorch**: https://pytorch.org/
- **Transformers**: https://huggingface.co/transformers/
- **PyMuPDF**: https://pymupdf.readthedocs.io/

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More tampering techniques
- Additional evaluation metrics
- Faster architectures (ViT, EfficientNet)
- Multi-page PDF support
- Cross-dataset evaluation

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size in config.py
BATCH_SIZE = 4  # or 2
```

### Slow Training

```bash
# Use GPU if available
DEVICE = 'cuda'

# Enable AMP (already enabled in train.py)
# Reduces memory and speeds up 2-3×
```

### Poor Results

1. **Check data quality**: Are PDFs readable?
2. **Increase training data**: Add more original PDFs
3. **Train longer**: Increase NUM_EPOCHS
4. **Adjust hyperparameters**: Tune learning rate, loss weights

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the author.

---

**Built with ❤️ for PDF Forensics Research**