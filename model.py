import torch
import torch.nn as nn
from transformers import DistilBertModel, AutoTokenizer
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Dict, Tuple


class MultiModalHashingModel(nn.Module):
    """
    Deep Supervised Hashing model for document forensics.
    
    Architecture:
        Text: DistilBERT → 768-dim → 64-dim hash
        Image: EfficientNet-B0 → 1280-dim → 64-dim hash
        Fusion: Concatenate → 64-dim final hash
    """
    
    def __init__(self, hash_bits: int = 64):
        super(MultiModalHashingModel, self).__init__()
        self.hash_bits = hash_bits

        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_output_dim = self.text_encoder.config.hidden_size # 768

        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.image_output_dim = effnet.classifier[1].in_features # 1280
        effnet.classifier = nn.Identity() # Remove the final classifier
        self.image_encoder = effnet

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
        if 'token_type_ids' in text_input:
            text_input_for_bert = text_input.copy()
            del text_input_for_bert['token_type_ids']
        else:
            text_input_for_bert = text_input

        text_outputs = self.text_encoder(**text_input_for_bert, return_dict=True)
        text_embedding = text_outputs.last_hidden_state[:, 0, :] # [CLS] token
        text_hash = self.text_projection(text_embedding)

        # EfficientNet forward pass
        image_feature = self.image_encoder(img_input)
        # No need to view/flatten, Identity() output is already (B, 1280)
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
    return (1.0 - (hash1 * hash2)) / 2.0 * hash1.shape[1]


def cosine_similarity_hashing(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between hash codes."""
    return torch.nn.functional.cosine_similarity(hash1, hash2, dim=1)