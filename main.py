"""
Main Pipeline for Multimodal Deep Hashing PDF Forensics
Bachelor Thesis Implementation

Pipeline Stages:
    1. Data Acquisition (optional)
    2. Data Generation & Tampering
    3. Model Training
    4. Model Evaluation
"""

import os
import sys
from config import (
    DEVICE, ORIGINAL_PDF_DIR, TAMPERED_PDF_DIR, 
    RESULTS_DIR, SAMPLES_PER_ORIGINAL
)
from data_generator import generate_dataset_csv
from train import main_train
from evaluate import evaluate_model


def setup_directories() -> bool:
    """
    Ensures all necessary directories exist and validates setup.
    
    Returns:
        True if setup is successful, False otherwise
    """
    print(f"\n{'='*70}")
    print("Setting up project directories...")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(ORIGINAL_PDF_DIR, exist_ok=True)
    os.makedirs(TAMPERED_PDF_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"✅ Directory structure created:")
    print(f"   - {ORIGINAL_PDF_DIR}")
    print(f"   - {TAMPERED_PDF_DIR}")
    print(f"   - checkpoints/")
    print(f"   - {RESULTS_DIR}")
    
    # Validate original PDFs exist
    pdf_files = [f for f in os.listdir(ORIGINAL_PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n{'='*70}")
        print("⚠️  WARNING: No PDF files found!")
        print(f"{'='*70}")
        print(f"\nPlease add PDF files to: {ORIGINAL_PDF_DIR}")
        print("\nYou can:")
        print("  1. Manually copy PDF files to the directory")
        print("  2. Use the download_pdfs.py script (requires valid data source)")
        print("\nExample PDFs for testing:")
        print("  - Research papers from arXiv")
        print("  - Academic documents")
        print("  - Technical reports")
        print(f"\n{'='*70}\n")
        return False
    
    print(f"\n✅ Found {len(pdf_files)} PDF files in {ORIGINAL_PDF_DIR}")
    print(f"{'='*70}\n")
    return True


def print_banner():
    """Prints the project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       Multimodal Deep Hashing for PDF Forensics                     ║
║       Bachelor Thesis Implementation                                ║
║                                                                      ║
║       Detecting Document Tampering Using:                           ║
║         • Deep Supervised Hashing (DSH)                             ║
║         • BERT (Text Encoder)                                       ║
║         • ResNet (Image Encoder)                                    ║
║         • Multimodal Fusion                                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"Target Device: {DEVICE}")
    print(f"{'='*70}\n")


def main():
    """
    Main execution function for the complete pipeline.
    """
    # Print banner
    print_banner()
    
    # Setup and validation
    if not setup_directories():
        sys.exit(1)
    
    # User interaction for pipeline control
    print("Pipeline Options:")
    print("  [1] Full Pipeline (Data Generation → Training → Evaluation)")
    print("  [2] Generate Data Only")
    print("  [3] Train Model Only (requires existing dataset)")
    print("  [4] Evaluate Model Only (requires trained model)")
    print()
    
    choice = input("Select option [1-4] (press Enter for option 1): ").strip()
    if not choice:
        choice = "1"
    
    print()
    
    # Phase 1: Data Generation
    if choice in ["1", "2"]:
        print(f"{'='*70}")
        print("[Phase 1/3] Data Generation and Tampering")
        print(f"{'='*70}\n")
        print(f"Generating {SAMPLES_PER_ORIGINAL} tampered versions per original PDF...")
        print("Tampering techniques: ITI, Zero-width spaces, Metadata changes, etc.\n")
        
        try:
            generate_dataset_csv(num_samples_per_orig=SAMPLES_PER_ORIGINAL)
            print("✅ Data generation complete!\n")
        except Exception as e:
            print(f"❌ Error during data generation: {e}")
            sys.exit(1)
    
    if choice == "2":
        print("Data generation complete. Exiting.")
        return
    
    # Phase 2: Model Training
    if choice in ["1", "3"]:
        print(f"{'='*70}")
        print("[Phase 2/3] Model Training")
        print(f"{'='*70}\n")
        
        # Verify dataset exists
        from config import DATASET_CSV_PATH
        if not os.path.exists(DATASET_CSV_PATH):
            print(f"❌ Dataset not found: {DATASET_CSV_PATH}")
            print("Please run data generation first (option 1 or 2)")
            sys.exit(1)
        
        try:
            train_history = main_train()
            print("✅ Model training complete!\n")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    if choice == "3":
        print("Training complete. Exiting.")
        return
    
    # Phase 3: Model Evaluation
    if choice in ["1", "4"]:
        print(f"{'='*70}")
        print("[Phase 3/3] Model Evaluation")
        print(f"{'='*70}\n")
        
        # Verify model exists
        from config import MODEL_SAVE_PATH
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"❌ Trained model not found: {MODEL_SAVE_PATH}")
            print("Please run training first (option 1 or 3)")
            sys.exit(1)
        
        try:
            evaluate_model()
            print("✅ Evaluation complete!\n")
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎉 Pipeline Execution Complete!")
    print(f"{'='*70}")
    print("\nGenerated Outputs:")
    print(f"  • Dataset: {DATASET_CSV_PATH}")
    print(f"  • Trained Model: {MODEL_SAVE_PATH}")
    print(f"  • Results: {RESULTS_DIR}/")
    print("\nNext Steps:")
    print("  • Review evaluation metrics in results/evaluation_results.txt")
    print("  • Analyze visualizations in results/*.png")
    print("  • Fine-tune hyperparameters in config.py if needed")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)