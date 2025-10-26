"""
Multimodal Deep Hashing Model for PDF Forensics
Architecture: BERT (text) + ResNet (image) + Hash Projection
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Tuple


class MultiModalHashingModel(nn.Module):
    """
    Deep Supervised Hashing model for document forensics.
    
    Architecture:
        Text: BERT-base → 768-dim → 64-dim hash
        Image: ResNet-18 → 512-dim → 64-dim hash
        Fusion: Concatenate → 64-dim final hash
    """
    
    def __init__(self, hash_bits: int = 64):
        super(MultiModalHashingModel, self).__init__()
        self.hash_bits = hash_bits

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_output_dim = self.text_encoder.config.hidden_size

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_output_dim = 512

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_output_dim, hash_bits),
            nn.Tanh()
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_output_dim, hash_bits),
            nn.Tanh()
        )
        
        self.fusion_dim = hash_bits * 2
        self.hash_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, hash_bits),
            nn.Tanh()
        )

    def forward_single_doc(self, text_input: Dict[str, torch.Tensor], 
                           img_input: torch.Tensor) -> torch.Tensor:
        """Process single document and produce hash code."""
        text_outputs = self.text_encoder(**text_input, return_dict=True)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_hash = self.text_projection(text_embedding)

        image_feature = self.image_encoder(img_input)
        image_feature = image_feature.view(image_feature.size(0), -1)
        image_hash = self.image_projection(image_feature)

        combined_features = torch.cat((text_hash, image_hash), dim=1)
        final_hash = self.hash_layer(combined_features)
        
        return final_hash

    def forward(self, doc1_modalities: Tuple[Dict, torch.Tensor], 
                doc2_modalities: Tuple[Dict, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for document pair."""
        text1, img1 = doc1_modalities
        text2, img2 = doc2_modalities
        
        hash1 = self.forward_single_doc(text1, img1)
        hash2 = self.forward_single_doc(text2, img2)
        
        return hash1, hash2


def binarize_hash(hash_output: torch.Tensor) -> torch.Tensor:
    """Convert continuous hash to binary {-1, +1}."""
    return torch.sign(hash_output)


def hamming_distance(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Compute Hamming distance between binary hashes."""
    return (hash1 != hash2).sum(dim=1).float()


def cosine_similarity_hashing(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between hash codes."""
    return torch.nn.functional.cosine_similarity(hash1, hash2, dim=1)