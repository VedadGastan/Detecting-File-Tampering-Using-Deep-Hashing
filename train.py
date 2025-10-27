"""
Training Module with Mixed Precision and Severity Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from config import (
    LEARNING_RATE, MODEL_SAVE_PATH, NUM_EPOCHS, DEVICE, 
    HASH_BIT_LENGTH, BETA_QUANT, GAMMA_DIST
)
from model import MultiModalHashingModel
from dataset import get_dataloader


class DeepHashingLoss(nn.Module):
    """
    Deep Supervised Hashing Loss with class imbalance weighting.
    
    The loss applies a weight of 3.0 to the minority class (identity pairs, Label=0)
    to counteract the 1:5 ratio of identity to tampered data.
    """
    
    def __init__(self, L: int = HASH_BIT_LENGTH, beta: float = BETA_QUANT, 
                 gamma: float = GAMMA_DIST):
        super().__init__()
        self.L = L
        self.beta = beta
        self.gamma = gamma
        
        # Weight for the minority class (Label 0: Identity/Similar).
        # Calculated as: Total Samples / (Minority Samples * 2) = 5400 / (900 * 2) = 3.0
        self.positive_weight = 3.0 

    def forward(self, hash1: torch.Tensor, hash2: torch.Tensor, 
                target_label: torch.Tensor) -> tuple:
        """
        Args:
            hash1, hash2: Hash codes (B, L) in [-1, 1]
            target_label: Binary labels (B,) - 0=similar, 1=dissimilar
        Returns:
            (total_loss, sim_loss, quant_loss, dist_loss)
        """
        
        target_label = target_label.float()

        inner_product = (hash1 * hash2).sum(dim=1) / self.L
        
        target_similarity = 1.0 - 2.0 * target_label 
        
        sim_loss_unweighted = (target_similarity * inner_product).clamp(min=-1.0).add(1.0).log1p().neg()
        
        weight = self.positive_weight * (1.0 - target_label) + (1.0 * target_label)
        
        sim_loss = (weight * sim_loss_unweighted).mean()

        quant_loss = (torch.abs(hash1) - 1.0).abs().mean() + (torch.abs(hash2) - 1.0).abs().mean()
        
        dist_loss = (hash1.sum(dim=0).abs() / hash1.size(0)).mean() + (hash2.sum(dim=0).abs() / hash2.size(0)).mean()
        
        total_loss = sim_loss + self.beta * quant_loss + self.gamma * dist_loss

        return total_loss, sim_loss, quant_loss.mean(), dist_loss.mean()


def main_train():
    """Main training with mixed precision and gradient accumulation."""
    
    print(f"\n{'='*50}")
    print(f"Training on {DEVICE} | Hash: {HASH_BIT_LENGTH} bits")
    print(f"LR: {LEARNING_RATE} | Epochs: {NUM_EPOCHS}")
    print(f"{'='*50}\n")
    
    model = MultiModalHashingModel(hash_bits=HASH_BIT_LENGTH).to(DEVICE)
    train_loader = get_dataloader(is_train=True)
    test_loader = get_dataloader(is_train=False)
    criterion = DeepHashingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Mixed precision training for speed
    scaler = GradScaler() if DEVICE == 'cuda' else None
    use_amp = DEVICE == 'cuda'
    
    history = {'train_loss': [], 'test_loss': [], 'sim_loss': [], 
               'quant_loss': [], 'dist_loss': []}
    best_test_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # === TRAINING ===
        model.train()
        epoch_metrics = {'total': 0, 'sim': 0, 'quant': 0, 'dist': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for doc1, doc2, labels in pbar:
            doc1_text, doc1_img = doc1
            doc2_text, doc2_img = doc2
            
            doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
            doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
            doc1_img = doc1_img.to(DEVICE)
            doc2_img = doc2_img.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
                    total_loss, sim_loss, quant_loss, dist_loss = criterion(hash1, hash2, labels)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
                total_loss, sim_loss, quant_loss, dist_loss = criterion(hash1, hash2, labels)
                total_loss.backward()
                optimizer.step()
            
            epoch_metrics['total'] += total_loss.item()
            epoch_metrics['sim'] += sim_loss.item()
            epoch_metrics['quant'] += quant_loss.item()
            epoch_metrics['dist'] += dist_loss.item()
            
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
        
        avg_train = epoch_metrics['total'] / len(train_loader)
        avg_sim = epoch_metrics['sim'] / len(train_loader)
        avg_quant = epoch_metrics['quant'] / len(train_loader)
        avg_dist = epoch_metrics['dist'] / len(train_loader)
        
        # === VALIDATION ===
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for doc1, doc2, labels in test_loader:
                doc1_text, doc1_img = doc1
                doc2_text, doc2_img = doc2
                
                doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
                doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
                doc1_img = doc1_img.to(DEVICE)
                doc2_img = doc2_img.to(DEVICE)
                labels = labels.to(DEVICE)
                
                if use_amp:
                    with autocast():
                        hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
                        loss, _, _, _ = criterion(hash1, hash2, labels)
                else:
                    hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
                    loss, _, _, _ = criterion(hash1, hash2, labels)
                test_loss += loss.item()
        
        avg_test = test_loss / len(test_loader)
        scheduler.step()
        
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)
        history['sim_loss'].append(avg_sim)
        history['quant_loss'].append(avg_quant)
        history['dist_loss'].append(avg_dist)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f} Test={avg_test:.4f}")
        
        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': avg_test,
                'history': history
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved (Loss: {avg_test:.4f})")
    
    print(f"\n{'='*50}")
    print(f"Training Complete | Best Loss: {best_test_loss:.4f}")
    print(f"Model: {MODEL_SAVE_PATH}")
    print(f"{'='*50}\n")
    
    return history


if __name__ == '__main__':
    main_train()