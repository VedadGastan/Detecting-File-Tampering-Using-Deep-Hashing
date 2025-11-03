"""
Enhanced Training Loop with 3-Modality Architecture
Faster training through optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from config import (
    LEARNING_RATE, MODEL_SAVE_PATH, NUM_EPOCHS, DEVICE, 
    HASH_BIT_LENGTH, BETA_QUANT, GAMMA_DIST
)
from model import MultiModalHashingModel
from dataset import get_dataloader


class DeepHashingLoss(nn.Module):
    """Deep Supervised Hashing Loss with class imbalance weighting."""
    
    def __init__(self, L: int = HASH_BIT_LENGTH, beta: float = BETA_QUANT, 
                 gamma: float = GAMMA_DIST):
        super().__init__()
        self.L = L
        self.beta = beta
        self.gamma = gamma
        self.positive_weight = 3.0  # Weight for minority class

    def forward(self, hash1: torch.Tensor, hash2: torch.Tensor, 
                target_label: torch.Tensor) -> tuple:
        
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
    """Enhanced training with 3-modality architecture."""
    
    print(f"\n{'='*60}")
    print(f"ENHANCED TRAINING - 3 Modalities")
    print(f"Device: {DEVICE} | Hash: {HASH_BIT_LENGTH} bits")
    print(f"LR: {LEARNING_RATE} | Epochs: {NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    # Initialize model with structural features
    model = MultiModalHashingModel(
        hash_bits=HASH_BIT_LENGTH,
        structural_feature_dim=40
    ).to(DEVICE)
    
    train_loader = get_dataloader(is_train=True)
    test_loader = get_dataloader(is_train=False)
    criterion = DeepHashingLoss()
    
    # Optimizer with separate learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.text_encoder.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained
        {'params': model.image_encoder.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.structural_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.text_projection.parameters(), 'lr': LEARNING_RATE},
        {'params': model.image_projection.parameters(), 'lr': LEARNING_RATE},
        {'params': model.hash_layer.parameters(), 'lr': LEARNING_RATE},
        {'params': model.modality_attention.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Mixed precision for speed (if CUDA available)
    scaler = GradScaler() if DEVICE == 'cuda' else None
    use_amp = DEVICE == 'cuda'
    
    history = {'train_loss': [], 'test_loss': [], 'sim_loss': [], 
               'quant_loss': [], 'dist_loss': []}
    best_test_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # TRAINING
        model.train()
        epoch_metrics = {'total': 0, 'sim': 0, 'quant': 0, 'dist': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for doc1, doc2, labels in pbar:
            doc1_text, doc1_img, doc1_struct = doc1
            doc2_text, doc2_img, doc2_struct = doc2
            
            # Move to device
            doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
            doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
            doc1_img = doc1_img.to(DEVICE)
            doc2_img = doc2_img.to(DEVICE)
            doc1_struct = doc1_struct.to(DEVICE)  # NEW
            doc2_struct = doc2_struct.to(DEVICE)  # NEW
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    hash1, hash2 = model(
                        (doc1_text, doc1_img, doc1_struct), 
                        (doc2_text, doc2_img, doc2_struct)
                    )
                    total_loss, sim_loss, quant_loss, dist_loss = criterion(hash1, hash2, labels)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                hash1, hash2 = model(
                    (doc1_text, doc1_img, doc1_struct), 
                    (doc2_text, doc2_img, doc2_struct)
                )
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
        
        # VALIDATION
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for doc1, doc2, labels in test_loader:
                doc1_text, doc1_img, doc1_struct = doc1
                doc2_text, doc2_img, doc2_struct = doc2
                
                doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
                doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
                doc1_img = doc1_img.to(DEVICE)
                doc2_img = doc2_img.to(DEVICE)
                doc1_struct = doc1_struct.to(DEVICE)
                doc2_struct = doc2_struct.to(DEVICE)
                labels = labels.to(DEVICE)
                
                if use_amp:
                    with autocast():
                        hash1, hash2 = model(
                            (doc1_text, doc1_img, doc1_struct), 
                            (doc2_text, doc2_img, doc2_struct)
                        )
                        loss, _, _, _ = criterion(hash1, hash2, labels)
                else:
                    hash1, hash2 = model(
                        (doc1_text, doc1_img, doc1_struct), 
                        (doc2_text, doc2_img, doc2_struct)
                    )
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
            print(f"✓ Model saved (Loss: {avg_test:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training Complete | Best Loss: {best_test_loss:.4f}")
    print(f"Model: {MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")
    
    return history


if __name__ == '__main__':
    main_train()