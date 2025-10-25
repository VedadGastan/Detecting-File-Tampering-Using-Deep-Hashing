"""
Configuration file for Multimodal Deep Hashing PDF Forensics System
Bachelor Thesis Implementation
"""

import torch

# --- Dataset and I/O Paths ---
ORIGINAL_PDF_DIR = "data/original_pdfs"
TAMPERED_PDF_DIR = "data/tampered_pdfs"
DATASET_CSV_PATH = "data/forensics_dataset.csv"
MODEL_SAVE_PATH = "checkpoints/multimodal_dsh_model.pth"
RESULTS_DIR = "results"
MAX_SEQ_LENGTH = 128  # Max tokens for text input

# --- Model Hyperparameters ---
HASH_BIT_LENGTH = 64  # Length of the final binary hash code (64, 128, etc.)
COMMON_EMBEDDING_DIM = 512  # Intermediate dimension before hashing
IMAGE_SIZE = 224  # Input resolution for the CNN stream (ResNet standard)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 8  # Reduced from 32 for stability with small datasets
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

# --- Deep Hashing Loss Specifics ---
PAIRWISE_MARGIN = 1.0  # Margin for similarity/dissimilarity
BETA_QUANT = 0.01  # Weight for the Quantization Loss (J_quant)
GAMMA_DIST = 0.05  # Weight for the Distribution/Uniformity Loss

# --- Tampering Configuration ---
TAMPER_TYPES = [
    "invisible_text",  # Invisible Text Injection (ITI)
    "zero_width_space",  # Zero-width character injection
    "meta_change",  # Metadata modification
    "toc_removal",  # Table of Contents removal
    "line_artifact",  # Visual artifact injection
    "image_recompress"  # Image recompression
]
SAMPLES_PER_ORIGINAL = 3  # Number of tampered versions per original PDF

# --- Evaluation Configuration ---
TOP_K_VALUES = [10, 20, 50, 100]  # For Precision@K and Recall@K metrics