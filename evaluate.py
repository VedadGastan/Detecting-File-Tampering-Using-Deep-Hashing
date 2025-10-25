"""
Evaluation module for PDF Forensics Deep Hashing System
Computes retrieval metrics: MAP, Precision@K, Recall@K, F1-Score
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import (
    DEVICE, HASH_BIT_LENGTH, MODEL_SAVE_PATH, 
    RESULTS_DIR, TOP_K_VALUES
)
from model import MultiModalHashingModel, binarize_hash, hamming_distance
from dataset import get_full_dataloader


def compute_map(distances: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> float:
    """
    Computes Mean Average Precision (MAP) for retrieval.
    
    Args:
        distances: Hamming distances matrix (N_query, N_database)
        labels: Ground truth labels for database samples (N_database,)
        query_labels: Ground truth labels for query samples (N_query,)
        
    Returns:
        Mean Average Precision score
    """
    num_queries = distances.shape[0]
    average_precisions = []
    
    for i in range(num_queries):
        # Get distances and labels for this query
        query_distances = distances[i]
        query_label = query_labels[i]
        
        # Sort by distance (ascending - smaller distance = more similar)
        sorted_indices = np.argsort(query_distances)
        sorted_labels = labels[sorted_indices]
        
        # Compute relevant items (same label as query)
        relevant = (sorted_labels == query_label).astype(int)
        
        # Skip if no relevant items
        if relevant.sum() == 0:
            continue
        
        # Compute precision at each position
        cumsum_relevant = np.cumsum(relevant)
        positions = np.arange(1, len(relevant) + 1)
        precisions = cumsum_relevant / positions
        
        # Average precision for this query
        ap = (precisions * relevant).sum() / relevant.sum()
        average_precisions.append(ap)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_precision_recall_at_k(distances: np.ndarray, labels: np.ndarray, 
                                   query_labels: np.ndarray, k: int) -> tuple:
    """
    Computes Precision@K and Recall@K.
    
    Args:
        distances: Hamming distances matrix (N_query, N_database)
        labels: Ground truth labels for database samples
        query_labels: Ground truth labels for query samples
        k: Top-K threshold
        
    Returns:
        Tuple of (precision@k, recall@k)
    """
    num_queries = distances.shape[0]
    precisions = []
    recalls = []
    
    for i in range(num_queries):
        query_distances = distances[i]
        query_label = query_labels[i]
        
        # Get top-K nearest neighbors
        top_k_indices = np.argsort(query_distances)[:k]
        top_k_labels = labels[top_k_indices]
        
        # Compute relevant items in top-K
        relevant_in_top_k = (top_k_labels == query_label).sum()
        total_relevant = (labels == query_label).sum()
        
        # Precision@K and Recall@K
        precision_k = relevant_in_top_k / k if k > 0 else 0
        recall_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        precisions.append(precision_k)
        recalls.append(recall_k)
    
    return np.mean(precisions), np.mean(recalls)


def evaluate_model():
    """
    Comprehensive evaluation of the trained hashing model.
    """
    print(f"\n{'='*70}")
    print(f"  Evaluating Multimodal Deep Hashing Model")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model
    model = MultiModalHashingModel(hash_bits=HASH_BIT_LENGTH).to(DEVICE)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from: {MODEL_SAVE_PATH}")
    print(f"   Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best test loss: {checkpoint.get('test_loss', 'N/A'):.4f}\n")
    
    # Load data
    dataloader = get_full_dataloader(batch_size=16)
    
    # Extract hash codes and labels
    all_hash1 = []
    all_hash2 = []
    all_labels = []
    
    print("Extracting hash codes...")
    with torch.no_grad():
        for doc1, doc2, labels in tqdm(dataloader):
            doc1_text, doc1_img = doc1
            doc2_text, doc2_img = doc2
            
            doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
            doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
            doc1_img = doc1_img.to(DEVICE)
            doc2_img = doc2_img.to(DEVICE)
            
            hash1, hash2 = model((doc1_text, doc1_img), (doc2_text, doc2_img))
            
            # Binarize hashes
            hash1_bin = binarize_hash(hash1)
            hash2_bin = binarize_hash(hash2)
            
            all_hash1.append(hash1_bin.cpu())
            all_hash2.append(hash2_bin.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_hash1 = torch.cat(all_hash1, dim=0)
    all_hash2 = torch.cat(all_hash2, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"✅ Extracted {len(all_hash1)} hash code pairs\n")
    
    # Compute Hamming distances
    print("Computing Hamming distances...")
    num_samples = len(all_hash1)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    for i in tqdm(range(num_samples)):
        distances = hamming_distance(
            all_hash1[i].unsqueeze(0).expand(num_samples, -1),
            all_hash2
        ).numpy()
        distance_matrix[i] = distances
    
    # Evaluation metrics
    print(f"\n{'='*70}")
    print("Retrieval Performance Metrics:")
    print(f"{'='*70}\n")
    
    # Mean Average Precision
    map_score = compute_map(distance_matrix, all_labels, all_labels)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    
    # Precision@K and Recall@K
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    for k in TOP_K_VALUES:
        if k <= num_samples:
            precision_k, recall_k = compute_precision_recall_at_k(
                distance_matrix, all_labels, all_labels, k
            )
            print(f"Precision@{k:<13} {precision_k:.4f}")
            print(f"Recall@{k:<16} {recall_k:.4f}")
    
    # Binary classification metrics
    print(f"\n{'='*70}")
    print("Binary Classification Metrics:")
    print(f"{'='*70}\n")
    
    # Use a threshold on Hamming distance
    threshold = HASH_BIT_LENGTH * 0.5  # 50% of bits different
    predictions = (distance_matrix.diagonal() > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary', zero_division=0
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Similar  Tampered")
    print(f"Actual Similar    {cm[0,0]:<6}  {cm[0,1]:<6}")
    print(f"       Tampered   {cm[1,0]:<6}  {cm[1,1]:<6}")
    
    # Visualization
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Similar', 'Tampered'],
                yticklabels=['Similar', 'Tampered'])
    plt.title('Confusion Matrix - PDF Tampering Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix saved to: {cm_path}")
    
    # Plot Hamming distance distribution
    plt.figure(figsize=(10, 6))
    similar_distances = distance_matrix.diagonal()[all_labels == 0]
    tampered_distances = distance_matrix.diagonal()[all_labels == 1]
    
    plt.hist(similar_distances, bins=30, alpha=0.6, label='Similar Pairs', color='green')
    plt.hist(tampered_distances, bins=30, alpha=0.6, label='Tampered Pairs', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.1f})')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.title('Hamming Distance Distribution')
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, 'hamming_distance_distribution.png')
    plt.savefig(dist_path, dpi=300)
    print(f"✅ Distance distribution saved to: {dist_path}")
    
    # Plot training history if available
    if 'train_history' in checkpoint:
        history = checkpoint['train_history']
        
        plt.figure(figsize=(12, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history['total_loss'], label='Total Loss', linewidth=2)
        plt.plot(history['test_loss'], label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Component losses
        plt.subplot(1, 2, 2)
        plt.plot(history['sim_loss'], label='Similarity Loss', linewidth=2)
        plt.plot(history['quant_loss'], label='Quantization Loss', linewidth=2)
        plt.plot(history['dist_loss'], label='Distribution Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_path = os.path.join(RESULTS_DIR, 'training_curves.png')
        plt.savefig(loss_path, dpi=300)
        print(f"✅ Training curves saved to: {loss_path}")
    
    # Save results to text file
    results_text = f"""
{'='*70}
PDF Forensics Deep Hashing - Evaluation Results
{'='*70}

Model Configuration:
- Hash Length: {HASH_BIT_LENGTH} bits
- Device: {DEVICE}
- Checkpoint: {MODEL_SAVE_PATH}

Retrieval Metrics:
- Mean Average Precision (MAP): {map_score:.4f}

Precision@K and Recall@K:
"""
    for k in TOP_K_VALUES:
        if k <= num_samples:
            precision_k, recall_k = compute_precision_recall_at_k(
                distance_matrix, all_labels, all_labels, k
            )
            results_text += f"  - Precision@{k}: {precision_k:.4f}\n"
            results_text += f"  - Recall@{k}: {recall_k:.4f}\n"
    
    results_text += f"""
Binary Classification Metrics:
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}

Confusion Matrix:
              Predicted
            Similar  Tampered
Actual Similar    {cm[0,0]:<6}  {cm[0,1]:<6}
       Tampered   {cm[1,0]:<6}  {cm[1,1]:<6}

{'='*70}
"""
    
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(results_text)
    
    print(f"✅ Evaluation results saved to: {results_path}")
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    evaluate_model()