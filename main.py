"""
Main Pipeline for PDF Forensics Deep Hashing System
"""

import os
import sys
from config import DEVICE, ORIGINAL_PDF_DIR, SAMPLES_PER_ORIGINAL
from data_generator import generate_dataset_csv
from train import main_train
from evaluate import evaluate_model


def check_setup():
    """Validate setup and check for PDFs."""
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {ORIGINAL_PDF_DIR}")
        print("\nPlease add PDF documents to the directory and try again.")
        print("Recommended: 50-100 diverse academic/technical PDFs\n")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files")
    return True


def print_header():
    """Print system header."""
    print("\n" + "="*50)
    print("  PDF FORENSICS - Deep Hashing System")
    print(f"  Device: {DEVICE}")
    print("="*50 + "\n")


def main():
    """Main execution pipeline."""
    print_header()
    
    if not check_setup():
        sys.exit(1)
    
    print("\nPipeline Options:")
    print("  [1] Full Pipeline (Generate → Train → Evaluate)")
    print("  [2] Generate Data Only")
    print("  [3] Train Model Only")
    print("  [4] Evaluate Model Only")
    
    choice = input("\nSelect [1-4] (default=1): ").strip() or "1"
    print()
    
    # === DATA GENERATION ===
    if choice in ["1", "2"]:
        print("="*50)
        print("PHASE 1: Data Generation")
        print("="*50 + "\n")
        
        try:
            generate_dataset_csv(num_samples_per_orig=SAMPLES_PER_ORIGINAL)
        except Exception as e:
            print(f"\n❌ Data generation failed: {e}")
            sys.exit(1)
    
    if choice == "2":
        print("\n✓ Data generation complete\n")
        return
    
    # === TRAINING ===
    if choice in ["1", "3"]:
        from config import DATASET_CSV_PATH
        if not os.path.exists(DATASET_CSV_PATH):
            print(f"\n❌ Dataset not found. Run data generation first.\n")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("PHASE 2: Model Training")
        print("="*50 + "\n")
        
        try:
            main_train()
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    if choice == "3":
        print("\n✓ Training complete\n")
        return
    
    # === EVALUATION ===
    if choice in ["1", "4"]:
        from config import MODEL_SAVE_PATH
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"\n❌ Model not found. Run training first.\n")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("PHASE 3: Model Evaluation")
        print("="*50 + "\n")
        
        try:
            evaluate_model()
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # === SUMMARY ===
    if choice == "1":
        from config import DATASET_CSV_PATH, MODEL_SAVE_PATH, RESULTS_DIR
        print("\n" + "="*50)
        print("✓ Pipeline Complete")
        print("="*50)
        print(f"\nGenerated Files:")
        print(f"  • Dataset: {DATASET_CSV_PATH}")
        print(f"  • Model: {MODEL_SAVE_PATH}")
        print(f"  • Results: {RESULTS_DIR}/")
        print("\nNext: Review results/evaluation_results.txt\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)