"""
PyTorch Dataset and DataLoader for Multimodal PDF Forensics
Handles paired document loading with text and image modalities
"""

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from typing import Tuple, Dict
from config import (
    DATASET_CSV_PATH, IMAGE_SIZE, MAX_SEQ_LENGTH, 
    BATCH_SIZE, DEVICE, TRAIN_TEST_SPLIT
)
from data_generator import extract_page_modalities

# Initialize BERT tokenizer for text processing
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')

# Image preprocessing pipeline (ImageNet normalization)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ForensicsDataset(Dataset):
    """
    Dataset for paired PDF documents with multimodal features.
    
    Each sample consists of:
        - Original document (text + image)
        - Paired document (text + image) - either identical or tampered
        - Binary label (0=similar/authentic, 1=dissimilar/tampered)
    """
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Args:
            data_df: DataFrame with columns [Original_Path, Paired_Path, Label, Tamper_Type]
        """
        self.data_df = data_df
        print(f"Dataset initialized with {len(self.data_df)} samples")
        
    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict, torch.Tensor], 
                                               Tuple[Dict, torch.Tensor], 
                                               torch.Tensor]:
        """
        Retrieves a single training sample.
        
        Returns:
            doc1_modalities: (text_tokens_dict, image_tensor) for original document
            doc2_modalities: (text_tokens_dict, image_tensor) for paired document
            label_tensor: Binary label (0 or 1)
        """
        row = self.data_df.iloc[idx]
        orig_path = row['Original_Path']
        paired_path = row['Paired_Path']
        label = row['Label']

        # Extract modalities for both documents
        orig_text, orig_img = extract_page_modalities(orig_path)
        paired_text, paired_img = extract_page_modalities(paired_path)

        # Transform images to tensors
        orig_img_tensor = IMAGE_TRANSFORM(orig_img)
        paired_img_tensor = IMAGE_TRANSFORM(paired_img)
            
        # Tokenize text (returns dict with input_ids, attention_mask, etc.)
        orig_tokens = TOKENIZER(
            orig_text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
        
        paired_tokens = TOKENIZER(
            paired_text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
        
        # Create label tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Package modalities as tuples
        doc1_modalities = (orig_tokens, orig_img_tensor)
        doc2_modalities = (paired_tokens, paired_img_tensor)
        
        return doc1_modalities, doc2_modalities, label_tensor


def collate_fn(batch):
    """
    Custom collate function to handle batching of tokenized text dictionaries.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched doc1_modalities, doc2_modalities, labels
    """
    doc1_list, doc2_list, label_list = zip(*batch)
    
    # Separate text and images
    doc1_texts, doc1_imgs = zip(*doc1_list)
    doc2_texts, doc2_imgs = zip(*doc2_list)
    
    # Stack images into batches
    doc1_imgs_batch = torch.stack(doc1_imgs)
    doc2_imgs_batch = torch.stack(doc2_imgs)
    
    # Concatenate text token dictionaries
    doc1_text_batch = {
        key: torch.cat([d[key] for d in doc1_texts], dim=0)
        for key in doc1_texts[0].keys()
    }
    doc2_text_batch = {
        key: torch.cat([d[key] for d in doc2_texts], dim=0)
        for key in doc2_texts[0].keys()
    }
    
    # Stack labels
    labels_batch = torch.stack(label_list)
    
    return (doc1_text_batch, doc1_imgs_batch), (doc2_text_batch, doc2_imgs_batch), labels_batch


def get_dataloader(is_train: bool = True, batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Creates a DataLoader for training or testing.
    
    Args:
        is_train: If True, returns training set; if False, returns test set
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader object
    """
    # Load the full dataset
    data_df = pd.read_csv(DATASET_CSV_PATH)
    
    # Split into train/test sets
    train_size = int(len(data_df) * TRAIN_TEST_SPLIT)
    
    if is_train:
        subset_df = data_df.iloc[:train_size]
        shuffle = True
    else:
        subset_df = data_df.iloc[train_size:]
        shuffle = False
    
    dataset = ForensicsDataset(subset_df)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )


def get_full_dataloader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Creates a DataLoader for the entire dataset (for evaluation).
    
    Args:
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader object
    """
    data_df = pd.read_csv(DATASET_CSV_PATH)
    dataset = ForensicsDataset(data_df)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )