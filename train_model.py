import torch
import torch.optim as optim
import glob
import random
from tqdm import tqdm
from sae_core import DeepHashModel, PDFFeaturizer, Config, HashDatabase
from tamper_engine import TamperEngine

def criterion(h1, h2, label, margin=2.0):
    dist = torch.nn.functional.pairwise_distance(h1, h2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0), 2)
    return loss.mean()

def train():
    files = glob.glob(f"{Config.TRAIN_DIR}/*.pdf")
    model = DeepHashModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        pbar = tqdm(range(32), desc=f"Epoch {epoch}")
        
        for _ in pbar:
            f1, f2, labels = [], [], []
            for _ in range(16):
                path = random.choice(files) if files else None
                label = random.choice([0.0, 1.0])
                
                feat1 = PDFFeaturizer.process(path) if path else PDFFeaturizer.process(TamperEngine.create_dummy_pdf())
                if label == 1.0:
                    doc2, _ = TamperEngine.apply_attack(path) if path else (TamperEngine.create_dummy_pdf(), 1.0)
                    feat2 = PDFFeaturizer.process(doc2)
                else:
                    feat2 = feat1.clone()
                
                f1.append(feat1); f2.append(feat2); labels.append(label)

            h1, h2 = model(torch.stack(f1)), model(torch.stack(f2))
            loss = criterion(h1, h2, torch.tensor(labels))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), Config.MODEL_PATH)
    print("Training Complete.")