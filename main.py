import os
import sys
import torch

try:
    from sae_core import Config, HashDatabase
except ImportError as e:
    print(f"Error: Missing required files. {e}")
    sys.exit(1)

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    model_status = "TRAINED" if os.path.exists(Config.MODEL_PATH) else "NOT TRAINED"
    
    db = HashDatabase()
    db_size = len(db.db)
    
    print("PDF TAMPER DETECTION SYSTEM")
    print(f"Model Status: {model_status}")
    print(f"Hash Database: {db_size} entries")
    print()

def test_feature_extraction():
    from sae_core import PDFFeaturizer
    import glob
    
    files = glob.glob(os.path.join(Config.TRAIN_DIR, "*.pdf"))
    if not files:
        print("No PDF files found in train_data folder")
        return
    
    print("Testing feature extraction on first 3 files:")
    for i, file_path in enumerate(files[:3]):
        features = PDFFeaturizer.process(file_path)
        print(f"\n{os.path.basename(file_path)}:")
        print(f"Features: {features.tolist()}")
        print(f"Non-zero features: {torch.sum(features != 0).item()}/22")

def main():
    while True:
        print_header()
        print("1. Train Model")
        print("2. Analyze PDF")
        print("3. View Database")
        print("4. Clear Database")
        print("5. Test Feature Extraction")
        print("6. Run Accuracy Test") # Added new option
        print("7. Exit")              # Updated option number
        
        choice = input("\nSelect option [1-7]: ").strip()
        
        if choice == '1':
            try:
                from train_model import train
                train()
            except Exception as e:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            if not os.path.exists(Config.MODEL_PATH):
                print("Error: Model not trained. Run training first.")
                input("Press Enter to continue...")
                continue
                
            path = input("Enter PDF file path: ").strip()
            if os.path.exists(path):
                try:
                    from forensics_tool import analyze_file
                    analyze_file(path)
                except Exception as e:
                    print(f"Analysis error: {e}")
            else:
                print(f"File not found: {path}")
            
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            db = HashDatabase()
            if db.db:
                print(f"\nDatabase ({len(db.db)} entries):")
                for name, entry in list(db.db.items())[:10]:
                    hash_val = entry['hash']
                    if len(hash_val) > 16:
                        hash_display = hash_val[:16] + "..."
                    else:
                        hash_display = hash_val
                    print(f"{name[:40]:<40} {hash_display}")
                if len(db.db) > 10:
                    print(f"... and {len(db.db) - 10} more")
            else:
                print("Database is empty")
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            confirm = input("Clear ALL database entries? (yes/no): ")
            if confirm.lower() == 'yes':
                db = HashDatabase()
                db.clear()
                print("Database cleared")
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            test_feature_extraction()
            input("\nPress Enter to continue...")

        # New option to run the accuracy test
        elif choice == '6':
            if not os.path.exists(Config.MODEL_PATH):
                print("Error: Model not trained. Run training first.")
                input("Press Enter to continue...")
                continue

            try:
                from test_accuracy import run_accuracy_test
                run_accuracy_test()
            except Exception as e:
                print(f"Accuracy test error: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nPress Enter to continue...")
            
        elif choice == '7':
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)