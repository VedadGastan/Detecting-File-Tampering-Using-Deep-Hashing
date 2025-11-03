"""
Enhanced Model Architecture with Structural Features
Now processes: Text + Image + PDF Structure for comprehensive tampering detection
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Dict, Tuple


class MultiModalHashingModel(nn.Module):
    """
    Enhanced Deep Supervised Hashing with 3 modalities:
    - Text: DistilBERT
    - Image: EfficientNet-B0  
    - Structure: MLP on PDF forensic features
    """
    
    def __init__(self, hash_bits: int = 64, structural_feature_dim: int = 40):
        super(MultiModalHashingModel, self).__init__()
        self.hash_bits = hash_bits
        self.structural_feature_dim = structural_feature_dim

        # Text Encoder (lighter config for speed)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Enable gradient checkpointing for memory efficiency
        self.text_encoder.gradient_checkpointing_enable()
        self.text_output_dim = self.text_encoder.config.hidden_size  # 768

        # Image Encoder (keep efficient)
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.image_output_dim = effnet.classifier[1].in_features  # 1280
        effnet.classifier = nn.Identity()
        self.image_encoder = effnet
        
        # NEW: Structural Feature Encoder
        # Processes PDF forensic features (metadata, links, JS, annotations, etc.)
        self.structural_encoder = nn.Sequential(
            nn.Linear(structural_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, hash_bits),
            nn.Tanh()
        )

        # Individual projections for each modality
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_output_dim, hash_bits),
            nn.Tanh()
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_output_dim, hash_bits),
            nn.Tanh()
        )
        
        # Fusion layer (now combines 3 hash codes)
        self.fusion_dim = hash_bits * 3  # Text + Image + Structural
        self.hash_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, hash_bits * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hash_bits * 2, hash_bits),
            nn.Tanh()
        )
        
        # Attention mechanism for modality weighting
        self.modality_attention = nn.Sequential(
            nn.Linear(self.fusion_dim, 3),
            nn.Softmax(dim=1)
        )

    def forward_single_doc(self, text_input: Dict[str, torch.Tensor],
                           img_input: torch.Tensor,
                           structural_input: torch.Tensor) -> torch.Tensor:
        """Process single document with all three modalities."""
        
        # Text encoding
        if 'token_type_ids' in text_input:
            text_input_clean = {k: v for k, v in text_input.items() if k != 'token_type_ids'}
        else:
            text_input_clean = text_input
        
        text_outputs = self.text_encoder(**text_input_clean, return_dict=True)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS]
        text_hash = self.text_projection(text_embedding)

        # Image encoding
        image_feature = self.image_encoder(img_input)
        image_hash = self.image_projection(image_feature)
        
        # NEW: Structural encoding
        structural_hash = self.structural_encoder(structural_input)

        # Combine all three hashes
        combined_features = torch.cat([text_hash, image_hash, structural_hash], dim=1)
        
        # Apply attention-weighted fusion
        attention_weights = self.modality_attention(combined_features)
        
        # Weighted combination
        weighted_text = text_hash * attention_weights[:, 0:1]
        weighted_image = image_hash * attention_weights[:, 1:2]
        weighted_structural = structural_hash * attention_weights[:, 2:3]
        
        weighted_features = torch.cat([weighted_text, weighted_image, weighted_structural], dim=1)
        
        # Final hash
        final_hash = self.hash_layer(weighted_features)
        
        return final_hash

    def forward(self, doc1_modalities: Tuple[Dict, torch.Tensor, torch.Tensor], 
                doc2_modalities: Tuple[Dict, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for document pair with all modalities."""
        text1, img1, struct1 = doc1_modalities
        text2, img2, struct2 = doc2_modalities
        
        hash1 = self.forward_single_doc(text1, img1, struct1)
        hash2 = self.forward_single_doc(text2, img2, struct2)
        
        return hash1, hash2


def binarize_hash(hash_output: torch.Tensor) -> torch.Tensor:
    """Convert continuous hash to binary {-1, +1}."""
    return torch.sign(hash_output)


def hamming_distance(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Compute Hamming distance between binary hashes."""
    return (1.0 - (hash1 * hash2)) / 2.0 * hash1.shape[1]


def cosine_similarity_hashing(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between hash codes."""
    return torch.nn.functional.cosine_similarity(hash1, hash2, dim=1)