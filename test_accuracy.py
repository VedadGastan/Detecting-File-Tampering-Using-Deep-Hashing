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

def run_accuracy_test(num_samples=100):
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model not found at {Config.MODEL_PATH}")
        return

    test_pdf_files = glob.glob(f"{Config.TEST_DIR}/*.pdf")
    
    model = DeepHashModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    same_distances = []
    different_distances = []
    tp, tn, fp, fn = 0, 0, 0, 0
    
    pbar = tqdm(range(num_samples), desc="Testing Accuracy")
    
    for _ in pbar:
        try:
            # 1. Load Original
            if test_pdf_files and random.random() > 0.1:
                path = random.choice(test_pdf_files)
                doc_original = fitz.open(path)
            else:
                doc_original = TamperEngine.create_dummy_pdf()

            # 2. Hash Original
            feat_orig = PDFFeaturizer.process(doc_original)
            with torch.no_grad():
                hash_orig = model.get_hex_hash(feat_orig)
            
            # 3. Clone and Tamper
            # We use insert_pdf to make a fully independent editable copy
            doc_test = fitz.open()
            doc_test.insert_pdf(doc_original)
            
            is_tampered = random.choice([True, False])
            
            if is_tampered:
                # Apply attack
                TamperEngine.apply_attack_to_doc(doc_test)
            
            # 4. Hash Test Document
            feat_test = PDFFeaturizer.process(doc_test)
            with torch.no_grad():
                hash_test = model.get_hex_hash(feat_test)
                
            dist = calculate_hamming_distance(hash_orig, hash_test)
            
            # 5. Metrics
            is_detected = dist > Config.HAMMING_THRESHOLD
            
            if is_tampered:
                different_distances.append(dist)
                if is_detected: tp += 1
                else: fn += 1
            else:
                same_distances.append(dist)
                if not is_detected: tn += 1
                else: fp += 1
                
            doc_original.close()
            doc_test.close()
            
        except Exception as e:
            continue

    pbar.close()
    
    # Calculate Scores
    total = tp + tn + fp + fn
    if total == 0: return print("No samples processed.")
    
    acc = (tp + tn) / total * 100
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    avg_diff = sum(different_distances)/len(different_distances) if different_distances else 0
    avg_same = sum(same_distances)/len(same_distances) if same_distances else 0

    print(f"\n{'='*40}")
    print(f"DEEP HASH ACCURACY REPORT")
    print(f"{'='*40}")
    print(f"Accuracy:       {acc:.2f}%")
    print(f"F1 Score:       {f1:.2f}")
    print(f"Precision:      {prec:.2f}")
    print(f"Recall:         {rec:.2f}")
    print(f"{'-'*40}")
    print(f"True Positives: {tp} (Tampering Detected)")
    print(f"True Negatives: {tn} (Clean Verified)")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn} (Missed Tampering)")
    print(f"{'-'*40}")
    print(f"Avg Dist (Clean):    {avg_same:.2f} bits")
    print(f"Avg Dist (Tampered): {avg_diff:.2f} bits")
    print(f"Threshold:           {Config.HAMMING_THRESHOLD} bits")
    print(f"{'='*40}\n")