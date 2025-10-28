import os
import sys
import subprocess
from config import (
    DEVICE, ORIGINAL_PDF_DIR, SAMPLES_PER_ORIGINAL, DATASET_CSV_PATH, 
    MODEL_SAVE_PATH, RESULTS_DIR
)

# Import core functions (assuming they are defined in their respective files)
from data_generator import generate_dataset_csv
from train import main_train
from evaluate import evaluate_model

# --- NEW PATHS FOR OPTIMIZED PIPELINE ---
PREPROCESSED_DIR = "data/preprocessed"
PREPROCESSED_CSV_PATH = os.path.join(PREPROCESSED_DIR, "dataset_preprocessed.csv")


def print_header():
    """Print system header with hardware context."""
    print("\n" + "="*70)
    print(" PDF FORENSICS - Multimodal Deep Hashing System")
    print("="*70)
    print(f" STATUS: Running on {DEVICE.upper()}")
    if DEVICE == 'cpu':
        print(" NOTE: Using CPU/Optimized pipeline.")
    print(f" Source Directory: {ORIGINAL_PDF_DIR}\n")


def check_original_pdfs():
    """Validate setup and check for PDFs."""
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n[SETUP ERROR] No PDF files found in {ORIGINAL_PDF_DIR}")
        print(" Action: Please run 'python scrape_arxiv.py' to acquire documents.")
        return False
    
    print(f"[SETUP OK] Found {len(pdf_files)} original PDF files.")
    return True


def run_preprocessing():
    """Executes the external preprocessing script."""
    print("\n" + "-"*70)
    print("PHASE 2: Data Pre-processing (Optimizing for CPU Performance)")
    print(" Status: Rendering PDFs and saving assets to disk for faster training...")
    print("-"*70)
    
    try:
        # Call the external script
        subprocess.run([sys.executable, "preprocess_data.py"], check=True)
        
        if not os.path.exists(PREPROCESSED_CSV_PATH):
             raise FileNotFoundError(f"Expected file not created: {PREPROCESSED_CSV_PATH}")
             
        print("\n[SUCCESS] Data pre-processing complete.")
        print(f" Output: {PREPROCESSED_CSV_PATH}")
    except FileNotFoundError:
        print("\n[ERROR] 'preprocess_data.py' not found.")
        print(" Action: Please ensure 'preprocess_data.py' is in the main directory.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pre-processing failed. Check 'preprocess_data.py' output.")
        sys.exit(1)


def main():
    """Main execution pipeline."""
    print_header()
    
    if not check_original_pdfs():
        sys.exit(1)
    
    print("\nPipeline Options:")
    # UPDATED OPTIONS
    print(" [1] FULL RUN (Generate → Preprocess → Train → Evaluate)")
    print(" [2] Generate Raw Data Only (Pairing & Tampering)") # NEW DEDICATED OPTION
    print(" [3] Pre-process Data Only (Requires dataset.csv)")
    print(" [4] Train Model Only (Requires pre-processed data)")
    print(" [5] Evaluate Model Only (Requires trained model)")
    
    choice = input("\nSelect Option [1-5] (default=1): ").strip() or "1"
    
    # --- STEP 1: DATA GENERATION (PDFs to CSV) ---
    # Choice 1 (Full) and Choice 2 (Generate Only) both run this block
    if choice in ["1", "2"]:
        print("\n" + "="*70)
        print("PHASE 1: Raw Data Generation (Pairing & Tampering)")
        print("="*70)
        try:
            generate_dataset_csv(num_samples_per_orig=SAMPLES_PER_ORIGINAL)
            if not os.path.exists(DATASET_CSV_PATH):
                raise FileNotFoundError(f"Expected file not created: {DATASET_CSV_PATH}")
            print(f"\n[SUCCESS] Raw dataset created: {DATASET_CSV_PATH}")
        except Exception as e:
            print(f"\n[ERROR] Data generation failed. Details: {e}")
            sys.exit(1)
            
    if choice == "2": return # Exit after generation if this option was selected

    # --- STEP 2: PRE-PROCESSING (CSV to Optimized Image/Text files) ---
    # Choice 1 (Full) and Choice 3 (Preprocess Only) both run this block
    if choice in ["1", "3"]:
        if not os.path.exists(DATASET_CSV_PATH):
            print(f"\n[DEPENDENCY ERROR] Missing {DATASET_CSV_PATH}. Run Option 2 first.")
            sys.exit(1)
        run_preprocessing()
    
    if choice == "3": return # Exit after preprocessing if this option was selected

    # --- STEP 3: TRAINING ---
    # Choice 1 (Full) and Choice 4 (Train Only) both run this block
    if choice in ["1", "4"]:
        if not os.path.exists(PREPROCESSED_CSV_PATH):
            print(f"\n[DEPENDENCY ERROR] Missing pre-processed data. Run Option 3 first.")
            sys.exit(1)
            
        print("\n" + "="*70)
        print("PHASE 3: Model Training (CPU Optimized)")
        print("="*70)
        
        try:
            main_train()
            print(f"\n[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"\n[ERROR] Training failed. Details: {e}")
            sys.exit(1)
            
    if choice == "4": return # Exit after training if this option was selected

    # --- STEP 4: EVALUATION ---
    # Choice 1 (Full) and Choice 5 (Evaluate Only) both run this block
    if choice in ["1", "5"]:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"\n[DEPENDENCY ERROR] Missing trained model. Run Option 4 first.")
            sys.exit(1)
            
        print("\n" + "="*70)
        print("PHASE 4: Model Evaluation")
        print("="*70)
        
        try:
            evaluate_model()
            print(f"\n[SUCCESS] Evaluation complete. Results saved to {RESULTS_DIR}/")
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed. Details: {e}")
            sys.exit(1)
            
    if choice == "5": return # Exit after evaluation if this option was selected
    
    # --- FULL RUN SUMMARY ---
    if choice == "1":
        print("\n" + "="*70)
        print(" FULL PIPELINE COMPLETE")
        print("="*70)
        print(f" Final Model: {MODEL_SAVE_PATH}")
        print(f" Final Results: {RESULTS_DIR}/evaluation_results.txt")
        print("\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] An unexpected error occurred: {e}\n")
        sys.exit(1)