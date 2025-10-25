"""
Training module for Multimodal Deep Hashing PDF Forensics
Implements Deep Supervised Hashing (DSH) loss and training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
    Deep Supervised Hashing Loss combining:
        1. Similarity Loss (J_sim): Enforces similar/dissimilar relationships
        2. Quantization Loss (J_quant): Forces hash codes to be binary
        3. Distribution Loss (J_dist): Ensures balanced bit distribution (optional)
    """
    
    def __init__(self, L: int = HASH_BIT_LENGTH, beta: float = BETA_QUANT, 
                 gamma: float = GAMMA_DIST):
        """
        Args:
            L: Hash code length
            beta: Weight for quantization loss
            gamma: Weight for distribution loss
        """
        super().__init__()
        self.L = L
        self.beta = beta
        self.gamma = gamma

    def forward(self, hash1: torch.Tensor, hash2: torch.Tensor, 
                target_label: torch.Tensor) -> tuple:
        """
        Computes the total hashing loss.
        
        Args:
            hash1: Hash codes for document 1 (B, L) in [-1, 1]
            hash2: Hash codes for document 2 (B, L) in [-1, 1]
            target_label: Binary labels (B,) - 0 for similar, 1 for dissimilar
            
        Returns:
            Tuple of (total_loss, sim_loss, quant_loss, dist_loss)
        """
        batch_size = hash1.size(0)
        
        # 1. Similarity Loss (using inner product as similarity measure)
        # Inner product normalized by hash length
        inner_product = (hash1 * hash2).sum(dim=1) / self.L
        
        # Target: 1 for similar pairs (label=0), -1 for dissimilar pairs (label=1)
        target_similarity = 1.0 - 2.0 * target_label
        
        # Logistic loss: encourages inner_product * target to be positive
        sim_loss = torch.log(1.0 + torch.exp(-target_similarity * inner_product)).mean()

        # 2. Quantization Loss (encourages hash codes to be in {-1, +1})
        # Penalizes deviation from binary values
        quant_loss1 = (torch.abs(hash1) - 1.0).pow(2).mean()
        quant_loss2 = (torch.abs(hash2) - 1.0).pow(2).mean()
        quant_loss = (quant_loss1 + quant_loss2) / 2.0

        # 3. Distribution Loss (encourages balanced bit distribution)
        # Each bit should be equally likely to be +1 or -1
        mean_hash = (hash1.mean(dim=0) + hash2.mean(dim=0)) / 2.0
        dist_loss = mean_hash.pow(2).mean()
        
        # Total loss
        total_loss = sim_loss + self.beta * quant_loss + self.gamma * dist_loss
        
        return total_loss, sim_loss, quant_loss, dist_loss


def main_train():
    """
    Main training function for the multimodal deep hashing model.
    """
    
    print(f"\n{'='*70}")
    print(f"  Training Multimodal Deep Hashing Model")
    print(f"  Device: {DEVICE}")
    print(f"  Hash Length: {HASH_BIT_LENGTH} bits")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {get_dataloader(is_train=True).batch_size}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"{'='*70}\n")
    
    # Initialize model, loss, and optimizer
    model = MultiModalHashingModel(hash_bits=HASH_BIT_LENGTH).to(DEVICE)
    train_dataloader = get_dataloader(is_train=True)
    test_dataloader = get_dataloader(is_train=False)
    criterion = DeepHashingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    train_history = {
        'total_loss': [], 'sim_loss': [], 'quant_loss': [], 'dist_loss': [],
        'test_loss': []
    }
    
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_metrics = {'total': 0, 'sim': 0, 'quant': 0, 'dist': 0}
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_idx, (doc1, doc2, labels) in enumerate(train_pbar):
            # Move data to device
            doc1_text, doc1_img = doc1
            doc2_text, doc2_img = doc2
            
            doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
            doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
            doc1_img = doc1_img.to(DEVICE)
            doc2_img = doc2_img.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
            
            # Compute loss
            total_loss, sim_loss, quant_loss, dist_loss = criterion(hash1, hash2, labels)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_metrics['total'] += total_loss.item()
            epoch_metrics['sim'] += sim_loss.item()
            epoch_metrics['quant'] += quant_loss.item()
            epoch_metrics['dist'] += dist_loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Sim': f"{sim_loss.item():.4f}",
                'Quant': f"{quant_loss.item():.4f}"
            })
        
        # Calculate average training metrics
        num_batches = len(train_dataloader)
        avg_total = epoch_metrics['total'] / num_batches
        avg_sim = epoch_metrics['sim'] / num_batches
        avg_quant = epoch_metrics['quant'] / num_batches
        avg_dist = epoch_metrics['dist'] / num_batches
        
        # Validation phase
        model.eval()
        test_loss_total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Test]")
            for doc1, doc2, labels in test_pbar:
                doc1_text, doc1_img = doc1
                doc2_text, doc2_img = doc2
                
                doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
                doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
                doc1_img = doc1_img.to(DEVICE)
                doc2_img = doc2_img.to(DEVICE)
                labels = labels.to(DEVICE)
                
                hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
                total_loss, _, _, _ = criterion(hash1, hash2, labels)
                test_loss_total += total_loss.item()
                
                test_pbar.set_postfix({'Test Loss': f"{total_loss.item():.4f}"})
        
        avg_test_loss = test_loss_total / len(test_dataloader)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Save history
        train_history['total_loss'].append(avg_total)
        train_history['sim_loss'].append(avg_sim)
        train_history['quant_loss'].append(avg_quant)
        train_history['dist_loss'].append(avg_dist)
        train_history['test_loss'].append(avg_test_loss)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {avg_total:.4f} (Sim: {avg_sim:.4f}, Quant: {avg_quant:.4f}, Dist: {avg_dist:.4f})")
        print(f"  Test Loss:  {avg_test_loss:.4f}")
        print(f"{'='*70}\n")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_total,
                'test_loss': avg_test_loss,
                'train_history': train_history
            }, MODEL_SAVE_PATH)
            print(f"✅ Best model saved (Test Loss: {avg_test_loss:.4f})\n")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"{'='*70}\n")
    
    return train_history


if __name__ == '__main__':
    main_train()