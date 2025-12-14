import torch
import torch.optim as optim
import glob
import random
import os
from tqdm import tqdm
from sae_core import DeepHashModel, PDFFeaturizer, Config, HashDatabase

def contrastive_loss(h1, h2, label, margin=1.0):
    dist = torch.norm(h1 - h2, p=2, dim=1)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0), 2)
    return loss.mean()

def train():
    os.makedirs(Config.TRAIN_DIR, exist_ok=True)
    files = glob.glob(os.path.join(Config.TRAIN_DIR, "*.pdf"))
    
    if not files:
        print(f"No PDF files found in {Config.TRAIN_DIR}")
        print("Please add PDF files to the train_data folder and try again.")
        return
    
    print(f"Found {len(files)} PDF files for training")
    
    model = DeepHashModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        processed_batches = 0
        
        pbar = tqdm(range(50), desc=f"Epoch {epoch+1}/5")
        for batch_idx in pbar:
            batch_features1 = []
            batch_features2 = []
            batch_labels = []
            
            for _ in range(8):
                try:
                    path = random.choice(files)
                    feat1 = PDFFeaturizer.process(path)
                    
                    if torch.all(feat1 == 0):
                        continue
                    
                    label = random.choice([0.0, 1.0])
                    
                    if label == 1.0:
                        import fitz
                        try:
                            doc = fitz.open(path)
                            if doc.page_count > 0:
                                page = doc[0]
                                if random.random() > 0.5:
                                    page.insert_text((100, 100), f"TAMPERED_{random.randint(1000, 9999)}")
                                else:
                                    page.draw_rect(fitz.Rect(50, 50, 150, 100), color=(1, 0, 0), width=2)
                            feat2 = PDFFeaturizer.process(doc)
                            doc.close()
                        except Exception as e:
                            feat2 = feat1.clone()
                    else:
                        feat2 = feat1.clone()
                    
                    batch_features1.append(feat1)
                    batch_features2.append(feat2)
                    batch_labels.append(label)
                except Exception as e:
                    continue
            
            if len(batch_features1) < 2:
                continue
                
            try:
                features1 = torch.stack(batch_features1)
                features2 = torch.stack(batch_features2)
                labels = torch.FloatTensor(batch_labels)
                
                hash1 = model(features1)
                hash2 = model(features2)
                
                loss = contrastive_loss(hash1, hash2, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                processed_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                continue
        
        if processed_batches > 0:
            avg_loss = total_loss / processed_batches
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: No valid batches processed")
    
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")
    
    print("\nGenerating hashes for training files...")
    model.eval()
    db = HashDatabase()
    
    with torch.no_grad():
        for file_path in tqdm(files, desc="Processing"):
            try:
                fname = os.path.basename(file_path)
                features = PDFFeaturizer.process(file_path)
                
                if torch.all(features == 0):
                    print(f"Skipped {fname}: Failed to extract features")
                    continue
                    
                hash_val = model.get_hex_hash(features)
                if hash_val and not hash_val.startswith("00000000"):
                    db.add(fname, hash_val, features)
                    print(f"Added {fname}: {hash_val[:16]}...")
                else:
                    print(f"Warning: {fname} generated zero hash")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Database updated with {len(db.db)} entries")