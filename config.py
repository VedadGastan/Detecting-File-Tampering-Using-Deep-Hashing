"""
Configuration File for PDF Forensics Deep Hashing System
"""

import torch
import os

# === DEVICE CONFIGURATION ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === MODEL CONFIGURATION ===
HASH_BIT_LENGTH = 64
MAX_SEQ_LENGTH = 512
IMAGE_SIZE = 224

# === TRAINING CONFIGURATION ===
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.8

# === LOSS FUNCTION WEIGHTS ===
BETA_QUANT = 0.1     # Quantization loss weight
GAMMA_DIST = 0.01    # Distribution loss weight

# === DATA PATHS ===
ORIGINAL_PDF_DIR = "data/original_pdfs"
TAMPERED_PDF_DIR = "data/tampered_pdfs"
DATASET_CSV_PATH = "data/dataset.csv"

# === MODEL CHECKPOINTS ===
MODEL_SAVE_PATH = "checkpoints/best_model.pth"

# === RESULTS ===
RESULTS_DIR = "results"

# === DATA GENERATION ===
SAMPLES_PER_ORIGINAL = 5  # Number of tampered versions per original PDF

# All 15 tampering techniques
TAMPER_TYPES = [
    # HIGH SEVERITY
    "invisible_text",
    "zero_width_space",
    "javascript_injection",
    "link_manipulation",
    
    # MEDIUM SEVERITY
    "font_substitution",
    "page_rotation",
    "watermark_injection",
    "line_artifact",
    "image_recompress",
    "annotation_injection",
    
    # LOW SEVERITY
    "meta_change",
    "toc_removal",
    "bookmark_manipulation",
    "encryption_metadata",
    
    # CRITICAL SEVERITY
    "content_stream_reorder"
]

# === EVALUATION CONFIGURATION ===
TOP_K_VALUES = [1, 5, 10, 20]

# === DIRECTORY SETUP ===
os.makedirs(ORIGINAL_PDF_DIR, exist_ok=True)
os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)