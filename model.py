"""
Multimodal Deep Hashing Model for PDF Forensics
Combines BERT (text) and ResNet (image) encoders with hash projection layers
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Tuple


class MultiModalHashingModel(nn.Module):
    """
    Deep Supervised Hashing model for multimodal document forensics.
    
    Architecture:
        1. Text Encoder: BERT-base-uncased
        2. Image Encoder: ResNet-18
        3. Projection heads: Map to hash space
        4. Fusion layer: Combines modalities and produces final hash
    """
    
    def __init__(self, hash_bits: int = 64):
        """
        Args:
            hash_bits: Length of the binary hash code
        """
        super(MultiModalHashingModel, self).__init__()
        self.hash_bits = hash_bits

        # Text Encoder: BERT
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_output_dim = self.text_encoder.config.hidden_size  # 768

        # Image Encoder: ResNet-18 (pre-trained on ImageNet)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_output_dim = 512

        # Modality-specific projection heads
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_output_dim, hash_bits),
            nn.Tanh()
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_output_dim, hash_bits),
            nn.Tanh()
        )
        
        # Fusion layer: Combines text and image hash codes
        self.fusion_dim = hash_bits * 2
        self.hash_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, hash_bits),
            nn.Tanh()  # Tanh for soft binarization in [-1, 1]
        )

    def forward_single_doc(self, text_input: Dict[str, torch.Tensor], 
                           img_input: torch.Tensor) -> torch.Tensor:
        """
        Processes a single document's modalities and produces a hash code.
        
        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask' (B, L)
            img_input: Image tensor (B, 3, H, W)
            
        Returns:
            Hash code tensor (B, hash_bits)
        """
        # Text processing
        text_outputs = self.text_encoder(**text_input, return_dict=True)
        # Use [CLS] token representation
        text_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_hash = self.text_projection(text_embedding)

        # Image processing
        image_feature = self.image_encoder(img_input)
        image_feature = image_feature.view(image_feature.size(0), -1)
        image_hash = self.image_projection(image_feature)

        # Fusion
        combined_features = torch.cat((text_hash, image_hash), dim=1)
        final_hash = self.hash_layer(combined_features)
        
        return final_hash

    def forward(self, doc1_modalities: Tuple[Dict, torch.Tensor], 
                doc2_modalities: Tuple[Dict, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of documents.
        
        Args:
            doc1_modalities: Tuple of (text_dict, image_tensor) for document 1
            doc2_modalities: Tuple of (text_dict, image_tensor) for document 2
            
        Returns:
            Tuple of (hash1, hash2) - hash codes for both documents
        """
        text1, img1 = doc1_modalities
        text2, img2 = doc2_modalities
        
        hash1 = self.forward_single_doc(text1, img1)
        hash2 = self.forward_single_doc(text2, img2)
        
        return hash1, hash2


def binarize_hash(hash_output: torch.Tensor) -> torch.Tensor:
    """
    Converts continuous hash codes to binary codes.
    
    Args:
        hash_output: Continuous hash in [-1, 1]
        
    Returns:
        Binary hash in {-1, 1}
    """
    return torch.sign(hash_output)


def hamming_distance(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """
    Computes Hamming distance between binary hash codes.
    
    Args:
        hash1: Binary hash code (B, L)
        hash2: Binary hash code (B, L)
        
    Returns:
        Hamming distance (B,)
    """
    return (hash1 != hash2).sum(dim=1).float()


def cosine_similarity_hashing(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between hash codes (alternative to Hamming).
    
    Args:
        hash1: Hash code (B, L)
        hash2: Hash code (B, L)
        
    Returns:
        Cosine similarity (B,)
    """
    return torch.nn.functional.cosine_similarity(hash1, hash2, dim=1)