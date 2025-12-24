import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc
from utils import binarize, hamming


# --------------------------------------------------
# Hash extraction
# --------------------------------------------------

def extract_all_hashes(model, directory, transform, device):
    hashes = {}
    model.eval()
    with torch.no_grad():
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            hashes[f] = binarize(model(img)).cpu().numpy()[0]
    return hashes


# --------------------------------------------------
# Evaluation core
# --------------------------------------------------

def evaluate(
    model,
    orig_dir,
    tamp_dir,
    transform,
    device,
    impostor_samples=5,
    verbose=True
):
    orig_hashes = extract_all_hashes(model, orig_dir, transform, device)
    tamp_hashes = extract_all_hashes(model, tamp_dir, transform, device)

    y = []
    s = []

    orig_keys = list(orig_hashes.keys())

    # --- Pair generation ---
    for o_name, o_hash in orig_hashes.items():
        base = os.path.splitext(o_name)[0]

        # Genuine: original vs its tampered versions
        for t_name, t_hash in tamp_hashes.items():
            if t_name.startswith(base):
                s.append(hamming(o_hash[None, :], t_hash[None, :])[0])
                y.append(1)

        # Impostors: random other originals
        impostors = np.random.choice(
            [k for k in orig_keys if k != o_name],
            size=min(impostor_samples, len(orig_keys) - 1),
            replace=False
        )

        for imp in impostors:
            s.append(hamming(o_hash[None, :], orig_hashes[imp][None, :])[0])
            y.append(0)

    y = np.array(y)
    s = np.array(s)

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    genuine = s[y == 1]
    impostor = s[y == 0]

    # Distance statistics
    stats = {
        "genuine_mean": float(genuine.mean()),
        "genuine_std": float(genuine.std()),
        "impostor_mean": float(impostor.mean()),
        "impostor_std": float(impostor.std()),
        "separation": float(impostor.mean() - genuine.mean()),
    }

    # ROC + AUC + EER
    fpr, tpr, thresholds = roc_curve(y, -s)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    # Hash collapse indicators
    all_hashes = np.stack(list(orig_hashes.values()))
    unique_hashes = np.unique(all_hashes, axis=0).shape[0]
    bit_entropy = np.mean(
        -(
            np.mean(all_hashes, axis=0) * np.log2(np.mean(all_hashes, axis=0) + 1e-8) +
            (1 - np.mean(all_hashes, axis=0)) * np.log2(1 - np.mean(all_hashes, axis=0) + 1e-8)
        )
    )

    diagnostics = {
        "num_originals": len(orig_hashes),
        "num_tampered": len(tamp_hashes),
        "num_pairs": len(s),
        "auc": float(roc_auc),
        "eer": eer,
        "unique_hashes": int(unique_hashes),
        "hash_entropy": float(bit_entropy),
        **stats
    }

    # --------------------------------------------------
    # Interpretive output
    # --------------------------------------------------

    if verbose:
        print("\n=== Evaluation Summary ===")
        for k, v in diagnostics.items():
            print(f"{k:>20}: {v}")

        print("\n=== Interpretation ===")

        if stats["separation"] <= 0:
            print("⚠ No separation: impostor distances are not larger than genuine.")
        elif stats["separation"] < 5:
            print("⚠ Weak separation: hashes overlap heavily.")
        else:
            print("✓ Meaningful distance separation detected.")

        if unique_hashes < 0.2 * len(orig_hashes):
            print("⚠ Hash collapse likely: many images share identical hashes.")

        if bit_entropy < 0.2:
            print("⚠ Low bit entropy: hash bits are mostly constant.")

        if roc_auc < 0.6:
            print("⚠ AUC indicates near-random verification performance.")
        elif roc_auc < 0.8:
            print("• Moderate verification performance.")
        else:
            print("✓ Strong verification performance.")

    return diagnostics
