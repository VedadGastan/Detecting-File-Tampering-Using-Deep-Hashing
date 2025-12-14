import torch
import os
import glob
import random
import fitz
from tqdm import tqdm
from sae_core import DeepHashModel, PDFFeaturizer, Config, HashDatabase
from tamper_engine import TamperEngine

def calculate_hamming_distance(hash1, hash2):
    bits1 = [int(b) for b in bin(int(hash1, 16))[2:].zfill(Config.HASH_BITS)]
    bits2 = [int(b) for b in bin(int(hash2, 16))[2:].zfill(Config.HASH_BITS)]
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

def run_accuracy_test(num_samples=500):
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model not found at {Config.MODEL_PATH}")
        return

    test_pdf_files = glob.glob(f"{Config.TEST_DIR}/*.pdf")
    
    model = DeepHashModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    same_distances = []
    different_distances = []
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    
    # Use tqdm for a clean progress bar, disable internal prints
    pbar = tqdm(range(num_samples), desc="Testing", unit="samples")
    
    for _ in pbar:
        try:
            # 1. Get Original Document (from disk or synthetic)
            if test_pdf_files and random.random() > 0.1:
                pdf_path = random.choice(test_pdf_files)
                doc_original = fitz.open(pdf_path)
            else:
                doc_original = TamperEngine.create_dummy_pdf()
            
            # 2. Prepare features and hash for the original state
            features_original = PDFFeaturizer.process(doc_original)
            with torch.no_grad():
                hash_original = model.get_hex_hash(features_original)
            
            is_tampered = random.random() > 0.5
            
            # 3. Create a disposable copy of the document for testing
            # FIX: We create the clone silently. We read bytes into a variable first.
            # We use "pdf" as the first arg to ensure fitz treats it as a PDF stream.
            original_bytes = doc_original.tobytes()
            doc_clone = fitz.open("pdf", original_bytes)
            
            if is_tampered:
                # Apply attack to the disposable clone
                TamperEngine.apply_attack_to_doc(doc_clone)
            
            features_test = PDFFeaturizer.process(doc_clone)
            
            with torch.no_grad():
                hash_test = model.get_hex_hash(features_test)
            
            hamming_dist = calculate_hamming_distance(hash_original, hash_test)
            
            # --- Metrics Calculation ---
            if is_tampered:
                different_distances.append(hamming_dist)
            else:
                same_distances.append(hamming_dist)
            
            detected_as_tampered = hamming_dist > Config.HAMMING_THRESHOLD
            
            if is_tampered and detected_as_tampered:
                true_positives += 1
            elif not is_tampered and not detected_as_tampered:
                true_negatives += 1
            elif not is_tampered and detected_as_tampered:
                false_positives += 1
            elif is_tampered and not detected_as_tampered:
                false_negatives += 1
            
            # Clean up resources
            doc_original.close()
            doc_clone.close()
            
        except Exception as e:
            # In case of any fitz errors, just continue to next sample
            continue
    
    pbar.close()
    
    # Avoid division by zero
    total = true_positives + true_negatives + false_positives + false_negatives
    if total == 0:
        print("\nNo samples were processed successfully.")
        return

    accuracy = (true_positives + true_negatives) / total * 100
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n" + "="*40)
    print(f"DEEP HASH EVALUATION RESULTS")
    print(f"="*40)
    print(f"Samples Tested: {total}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"Precision:      {precision:.2f}")
    print(f"Recall:         {recall:.2f}")
    print(f"F1 Score:       {f1:.2f}")
    print(f"-"*40)
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"-"*40)
    
    if same_distances:
        avg_same = sum(same_distances) / len(same_distances)
        print(f"Avg Dist (Untampered): {avg_same:.2f} bits")
    
    if different_distances:
        avg_diff = sum(different_distances) / len(different_distances)
        print(f"Avg Dist (Tampered):   {avg_diff:.2f} bits")
    
    print(f"Threshold Used:        {Config.HAMMING_THRESHOLD} bits")
    print(f"="*40 + "\n")