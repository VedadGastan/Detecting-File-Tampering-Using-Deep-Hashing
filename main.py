import os
import sys

try:
    from scraper import fetch_arxiv_pdfs
    from train_model import train
    from forensics_tool import analyze_file
    from test_accuracy import run_accuracy_test
    from sae_core import Config, HashDatabase
except ImportError as e:
    print(f"Error: Missing required files. {e}")
    sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    model_status = "TRAINED" if os.path.exists(Config.MODEL_PATH) else "NOT TRAINED"
    
    db = HashDatabase()
    db_size = len(db.db)
    
    print("DEEP HASHING PDF TAMPER DETECTION SYSTEM")
    print(f"Model Status: {model_status}")
    print(f"Hash Database: {db_size} entries")
    print()

def view_database():
    db = HashDatabase()
    
    if not db.db:
        print("Hash database is empty.")
        return
    
    print(f"\nHash Database ({len(db.db)} entries):\n")
    print(f"{'Filename':<50} {'Hash':<20}")
    print("-" * 72)
    
    for filename, entry in list(db.db.items())[:20]:
        hash_short = entry['hash'][:16] + "..."
        print(f"{filename:<50} {hash_short:<20}")
    
    if len(db.db) > 20:
        print(f"\n... and {len(db.db) - 20} more entries")

def main():
    while True:
        print_header()
        print("1. Download ArXiv Papers")
        print("2. Train Deep Hash Model")
        print("3. Test Hash Accuracy")
        print("4. Analyze PDF")
        print("5. View Hash Database")
        print("6. Clear Hash Database")
        print("7. Exit")
        
        choice = input("\nSelect option [1-7]: ").strip()
        
        if choice == '1':
            fetch_arxiv_pdfs()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            if not os.path.exists(Config.TRAIN_DIR) or not os.listdir(Config.TRAIN_DIR):
                print("Warning: No training data found. Using synthetic data.")
                confirm = input("Continue? (y/n): ")
                if confirm.lower() != 'y': 
                    continue
            
            train()
            input("\nPress Enter to continue...")

        elif choice == '3':
            if not os.path.exists(Config.MODEL_PATH):
                print("Error: Model not found. Train model first.")
            else:
                run_accuracy_test()
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            if not os.path.exists(Config.MODEL_PATH):
                print("Error: Model not trained. Run option 2 first.")
                input("Press Enter to continue...")
                continue
                
            path = input("Enter path to PDF: ").strip().replace('"', '').replace("'", "")
            
            if os.path.exists(path):
                analyze_file(path)
            else:
                print(f"Error: File not found: {path}")
            
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            view_database()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            confirm = input("Clear all hash database entries? (y/n): ")
            if confirm.lower() == 'y':
                db = HashDatabase()
                db.clear()
                print("Hash database cleared.")
            input("\nPress Enter to continue...")
            
        elif choice == '7':
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)