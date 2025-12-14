import torch
import sys
import os
from sae_core import DeepHashModel, PDFFeaturizer, Config, HashDatabase

def analyze_file(path):
    if not os.path.exists(Config.MODEL_PATH): return print("Model missing.")
    
    model = DeepHashModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    db = HashDatabase()
    features = PDFFeaturizer.process(path)
    hash_code = model.get_hex_hash(features)
    
    print(f"\nTarget: {os.path.basename(path)}")
    print(f"Hash: 0x{hash_code}")
    
    match_name, dist = db.find_match(hash_code)
    
    if match_name:
        status = "INTEGRITY_OK" if dist == 0 else "TAMPERED" if dist > Config.HAMMING_THRESHOLD else "MODIFIED"
        print(f"Match: {match_name} (Dist: {dist} bits)")
        print(f"Status: {status}")
        
        stored_f = torch.tensor(db.db[match_name]['features'])
        diff = torch.abs(features - stored_f)
        indices = torch.argsort(diff, descending=True)
        
        print("\nTop Feature Anomalies:")
        for i in indices[:3]:
            if diff[i] > 0.01:
                print(f"  - {PDFFeaturizer.FEATURE_NAMES[i]}: Δ{diff[i]:.4f}")