"""
Enhanced Evaluation with 3-Modality Architecture
Now accurately detects all tampering types
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import DEVICE, HASH_BIT_LENGTH, MODEL_SAVE_PATH, RESULTS_DIR, TOP_K_VALUES
from model import MultiModalHashingModel, binarize_hash, hamming_distance
from dataset import get_full_dataloader

PREPROCESSED_CSV_PATH = "data/preprocessed/dataset_preprocessed.csv"


def compute_map(distances: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> float:
    """Mean Average Precision (MAP) for retrieval."""
    num_queries = distances.shape[0]
    average_precisions = []
    
    for i in range(num_queries):
        query_distances = distances[i]
        query_label = query_labels[i]
        
        sorted_indices = np.argsort(query_distances)
        sorted_labels = labels[sorted_indices]
        relevant = (sorted_labels == query_label).astype(int)
        
        if relevant.sum() == 0:
            continue
        
        cumsum_relevant = np.cumsum(relevant)
        positions = np.arange(1, len(relevant) + 1)
        precisions = cumsum_relevant / positions
        ap = (precisions * relevant).sum() / relevant.sum()
        average_precisions.append(ap)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_precision_recall_at_k(distances: np.ndarray, labels: np.ndarray, 
                                   query_labels: np.ndarray, k: int) -> tuple:
    """Precision@K and Recall@K metrics."""
    num_queries = distances.shape[0]
    precisions, recalls = [], []
    
    for i in range(num_queries):
        query_distances = distances[i]
        query_label = query_labels[i]
        
        top_k_indices = np.argsort(query_distances)[:k]
        top_k_labels = labels[top_k_indices]
        
        relevant_in_top_k = (top_k_labels == query_label).sum()
        total_relevant = (labels == query_label).sum()
        
        precision_k = relevant_in_top_k / k if k > 0 else 0
        recall_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        precisions.append(precision_k)
        recalls.append(recall_k)
    
    return np.mean(precisions), np.mean(recalls)


def analyze_by_severity(distances: np.ndarray, data_df: pd.DataFrame) -> dict:
    """Analyze detection performance by tampering severity."""
    results = {}
    
    for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        severity_mask = data_df['Severity'] == severity
        if severity_mask.sum() == 0:
            continue
        
        severity_distances = distances[severity_mask]
        severity_labels = data_df[severity_mask]['Label'].values
        
        # Calculate mean distance for tampered vs similar
        tampered_mask = severity_labels == 1
        if tampered_mask.sum() > 0:
            mean_tampered = severity_distances[tampered_mask].mean()
            results[severity] = {
                'count': tampered_mask.sum(),
                'mean_distance': mean_tampered
            }
    
    return results


def evaluate_model():
    """Comprehensive evaluation with enhanced model."""
    
    print(f"\n{'='*60}")
    print(f"ENHANCED EVALUATION - 3 Modalities")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model
    model = MultiModalHashingModel(
        hash_bits=HASH_BIT_LENGTH,
        structural_feature_dim=40
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model (epoch {checkpoint.get('epoch', 'N/A')})\n")
    
    dataloader = get_full_dataloader(batch_size=16)
    
    # Load dataset info for severity analysis
    data_df = pd.read_csv(PREPROCESSED_CSV_PATH)
    
    # Extract hashes
    all_hash1, all_hash2, all_labels = [], [], []
    
    print("Extracting hash codes...")
    with torch.no_grad():
        for doc1, doc2, labels in tqdm(dataloader):
            doc1_text, doc1_img, doc1_struct = doc1
            doc2_text, doc2_img, doc2_struct = doc2
            
            doc1_text = {k: v.to(DEVICE) for k, v in doc1_text.items()}
            doc2_text = {k: v.to(DEVICE) for k, v in doc2_text.items()}
            doc1_img = doc1_img.to(DEVICE)
            doc2_img = doc2_img.to(DEVICE)
            doc1_struct = doc1_struct.to(DEVICE)
            doc2_struct = doc2_struct.to(DEVICE)
            
            hash1, hash2 = model(
                (doc1_text, doc1_img, doc1_struct), 
                (doc2_text, doc2_img, doc2_struct)
            )
            hash1_bin = binarize_hash(hash1)
            hash2_bin = binarize_hash(hash2)
            
            all_hash1.append(hash1_bin.cpu())
            all_hash2.append(hash2_bin.cpu())
            all_labels.append(labels.cpu())
    
    all_hash1 = torch.cat(all_hash1, dim=0)
    all_hash2 = torch.cat(all_hash2, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"Extracted {len(all_hash1)} pairs\n")
    
    # Compute distances
    print("Computing distance matrix...")
    num_samples = len(all_hash1)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    for i in tqdm(range(num_samples)):
        distances = hamming_distance(
            all_hash1[i].unsqueeze(0).expand(num_samples, -1),
            all_hash2
        ).numpy()
        distance_matrix[i] = distances
    
    # RETRIEVAL METRICS
    print(f"\n{'='*60}")
    print("RETRIEVAL METRICS")
    print(f"{'='*60}\n")
    
    map_score = compute_map(distance_matrix, all_labels, all_labels)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    
    print("\nPrecision@K and Recall@K:")
    for k in TOP_K_VALUES:
        if k <= num_samples:
            precision_k, recall_k = compute_precision_recall_at_k(
                distance_matrix, all_labels, all_labels, k
            )
            print(f"  K={k:3d}: P={precision_k:.4f} | R={recall_k:.4f}")

    # CLASSIFICATION METRICS
    print(f"\n{'='*60}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*60}\n")
    
    distances_diag = distance_matrix.diagonal()
    threshold = HASH_BIT_LENGTH * 0.5
    predictions = (distances_diag > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary', zero_division=0
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(all_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"        Pred: Similar  Tampered")
    print(f"Similar:     {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"Tampered:    {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    # SEVERITY ANALYSIS
    print(f"\n{'='*60}")
    print("DETECTION BY SEVERITY")
    print(f"{'='*60}\n")
    
    severity_results = analyze_by_severity(distances_diag, data_df)
    for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        if severity in severity_results:
            info = severity_results[severity]
            print(f"{severity:8s}: {info['count']:4d} samples | Avg Distance: {info['mean_distance']:.2f}")
    
    # ROC AND PR CURVES
    roc_fpr, roc_tpr, _ = roc_curve(all_labels, distances_diag)
    roc_auc = auc(roc_fpr, roc_tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(all_labels, distances_diag)
    avg_precision = average_precision_score(all_labels, distances_diag)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # STATISTICAL TESTS
    similar_dist = distances_diag[all_labels == 0]
    tampered_dist = distances_diag[all_labels == 1]
    
    statistic, p_value = stats.mannwhitneyu(similar_dist, tampered_dist, alternative='less')
    mean_diff = np.mean(similar_dist) - np.mean(tampered_dist)
    pooled_std = np.sqrt((np.std(similar_dist)**2 + np.std(tampered_dist)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Similar docs:  {np.mean(similar_dist):.2f} ± {np.std(similar_dist):.2f}")
    print(f"Tampered docs: {np.mean(tampered_dist):.2f} ± {np.std(tampered_dist):.2f}")
    print(f"Mann-Whitney U p-value: {p_value:.4e}")
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    
    # VISUALIZATIONS
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(roc_fpr, roc_tpr, label=f"ROC (AUC={roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Enhanced 3-Modality Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)
    print("✓ ROC curve saved")
    
    # Distance Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similar_dist, bins=30, alpha=0.6, label='Similar', color='green', density=True)
    plt.hist(tampered_dist, bins=30, alpha=0.6, label='Tampered', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.0f})')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Density')
    plt.title('Distance Distribution - Enhanced Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'distance_distribution.png'), dpi=300)
    print("✓ Distance distribution saved")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Similar', 'Tampered'],
                yticklabels=['Similar', 'Tampered'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300)
    print("✓ Confusion matrix saved")
    
    # Save results
    results_text = f"""
ENHANCED PDF FORENSICS EVALUATION
{'='*60}
Model: 3-Modality Architecture (Text + Image + Structure)

RETRIEVAL METRICS
- MAP: {map_score:.4f}
- ROC AUC: {roc_auc:.4f}
- Average Precision: {avg_precision:.4f}

CLASSIFICATION METRICS
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}

DETECTION BY SEVERITY
"""
    for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        if severity in severity_results:
            info = severity_results[severity]
            results_text += f"{severity}: {info['count']} samples, Avg Dist: {info['mean_distance']:.2f}\n"
    
    results_text += f"""
CONFUSION MATRIX
              Predicted
            Similar  Tampered
Similar     {cm[0,0]:6d}    {cm[0,1]:6d}
Tampered    {cm[1,0]:6d}    {cm[1,1]:6d}
"""
    
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(results_text)
    
    print(f"✓ Results saved to {results_path}")
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    evaluate_model()