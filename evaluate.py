import torch
import numpy as np
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


def compute_map(distances: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> float:
    """
    Mean Average Precision (MAP) for retrieval.
    """
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
    """
    Precision@K and Recall@K metrics.
    """
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


def compute_roc_metrics(distances: np.ndarray, labels: np.ndarray) -> dict:
    """
    ROC Curve and AUC (Area Under Curve).
    """
    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}


def compute_pr_metrics(distances: np.ndarray, labels: np.ndarray) -> dict:
    """
    Precision-Recall Curve and Average Precision.
    """
    precision, recall, thresholds = precision_recall_curve(labels, distances)
    avg_precision = average_precision_score(labels, distances)
    return {'precision': precision, 'recall': recall, 
            'thresholds': thresholds, 'avg_precision': avg_precision}


def compute_statistical_tests(similar_distances: np.ndarray, 
                              tampered_distances: np.ndarray) -> dict:
    """
    Statistical significance tests.
    """
    statistic, p_value = stats.mannwhitneyu(similar_distances, tampered_distances, 
                                           alternative='less')
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(similar_distances) - np.mean(tampered_distances)
    pooled_std = np.sqrt((np.std(similar_distances)**2 + np.std(tampered_distances)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'mann_whitney_u': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_similar': np.mean(similar_distances),
        'mean_tampered': np.mean(tampered_distances),
        'std_similar': np.std(similar_distances),
        'std_tampered': np.std(tampered_distances)
    }


def analyze_hash_quality(hashes: torch.Tensor) -> dict:
    """
    Hash code quality metrics.
    """
    binary_hashes = (hashes > 0).float()
    bit_means = binary_hashes.mean(dim=0)
    bit_balance = torch.abs(bit_means - 0.5).mean().item()
    bit_variance = binary_hashes.var(dim=0).mean().item()
    
    return {
        'bit_balance': bit_balance,
        'bit_variance': bit_variance,
        'mean_hamming_distance': hamming_distance(
            hashes[:len(hashes)//2], 
            hashes[len(hashes)//2:len(hashes)//2*2]
        ).mean().item() if len(hashes) > 1 else 0
    }


def evaluate_model():
    """Comprehensive scientific evaluation."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating on {DEVICE}")
    print(f"{'='*50}\n")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model = MultiModalHashingModel(hash_bits=HASH_BIT_LENGTH).to(DEVICE)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model (epoch {checkpoint.get('epoch', 'N/A')})\n")
    
    dataloader = get_full_dataloader(batch_size=16)
    
    # Extract hashes
    all_hash1, all_hash2, all_labels = [], [], []
    
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
    
    # EVALUATION METRICS
    print(f"\n{'='*50}")
    print("RETRIEVAL METRICS")
    print(f"{'='*50}\n")
    
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
    print(f"\n{'='*50}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*50}\n")
    
    threshold = HASH_BIT_LENGTH * 0.5
    predictions = (distance_matrix.diagonal() > threshold).astype(int)
    
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
    
    # ROC AND PR CURVES
    distances_diag = distance_matrix.diagonal()
    roc_metrics = compute_roc_metrics(distances_diag, all_labels)
    pr_metrics = compute_pr_metrics(distances_diag, all_labels)
    
    print(f"\nROC AUC: {roc_metrics['auc']:.4f}")
    print(f"Average Precision: {pr_metrics['avg_precision']:.4f}")
    
    # STATISTICAL TESTS
    print(f"\n{'='*50}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*50}\n")
    
    similar_dist = distances_diag[all_labels == 0]
    tampered_dist = distances_diag[all_labels == 1]
    
    stats_results = compute_statistical_tests(similar_dist, tampered_dist)
    print(f"Similar docs mean distance:  {stats_results['mean_similar']:.2f} ± {stats_results['std_similar']:.2f}")
    print(f"Tampered docs mean distance: {stats_results['mean_tampered']:.2f} ± {stats_results['std_tampered']:.2f}")
    print(f"Mann-Whitney U p-value: {stats_results['p_value']:.4e}")
    print(f"Effect size (Cohen's d): {stats_results['cohens_d']:.4f}")
    
    # HASH QUALITY
    print(f"\n{'='*50}")
    print("HASH CODE QUALITY")
    print(f"{'='*50}\n")
    
    hash_quality = analyze_hash_quality(all_hash1)
    print(f"Bit balance: {hash_quality['bit_balance']:.4f} (closer to 0 is better)")
    print(f"Bit variance: {hash_quality['bit_variance']:.4f} (higher is better)")
    
    # VISUALIZATIONS
    print(f"\n{'='*50}")
    print("Generating visualizations...")
    print(f"{'='*50}\n")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(roc_metrics['fpr'], roc_metrics['tpr'], 
             label=f"ROC (AUC={roc_metrics['auc']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - PDF Tampering Detection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)
    print("✓ ROC curve saved")
    
    # PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(pr_metrics['recall'], pr_metrics['precision'], 
             label=f"PR (AP={pr_metrics['avg_precision']:.3f})", linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve.png'), dpi=300)
    print("✓ PR curve saved")
    
    # Distance Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similar_dist, bins=30, alpha=0.6, label='Similar', color='green', density=True)
    plt.hist(tampered_dist, bins=30, alpha=0.6, label='Tampered', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.0f})')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Density')
    plt.title('Distance Distribution by Document Type')
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
PDF FORENSICS EVALUATION RESULTS
{'='*50}

RETRIEVAL METRICS
- MAP: {map_score:.4f}
- ROC AUC: {roc_metrics['auc']:.4f}
- Average Precision: {pr_metrics['avg_precision']:.4f}

PRECISION@K AND RECALL@K
"""
    for k in TOP_K_VALUES:
        if k <= num_samples:
            p_k, r_k = compute_precision_recall_at_k(distance_matrix, all_labels, all_labels, k)
            results_text += f"  K={k:3d}: Precision={p_k:.4f}, Recall={r_k:.4f}\n"
    
    results_text += f"""
CLASSIFICATION METRICS
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}

STATISTICAL ANALYSIS
- Similar mean: {stats_results['mean_similar']:.2f} ± {stats_results['std_similar']:.2f}
- Tampered mean: {stats_results['mean_tampered']:.2f} ± {stats_results['std_tampered']:.2f}
- Mann-Whitney U p-value: {stats_results['p_value']:.4e}
- Effect size (Cohen's d): {stats_results['cohens_d']:.4f}

HASH QUALITY
- Bit balance: {hash_quality['bit_balance']:.4f}
- Bit variance: {hash_quality['bit_variance']:.4f}

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
    print(f"\n{'='*50}")
    print("Evaluation Complete")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    evaluate_model()