"""
PyTorch Dataset for Multimodal PDF Forensics with Severity Support
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
    BATCH_SIZE, TRAIN_TEST_SPLIT
)
from data_generator import extract_page_modalities

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ForensicsDataset(Dataset):
    """Dataset for paired PDF documents with tampering labels and severity."""
    
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df
        print(f"Loaded {len(self.data_df)} samples")
        
    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict, torch.Tensor], 
                                               Tuple[Dict, torch.Tensor], 
                                               torch.Tensor]:
        row = self.data_df.iloc[idx]
        orig_path = row['Original_Path']
        paired_path = row['Paired_Path']
        label = row['Label']

        orig_text, orig_img = extract_page_modalities(orig_path)
        paired_text, paired_img = extract_page_modalities(paired_path)

        orig_img_tensor = IMAGE_TRANSFORM(orig_img)
        paired_img_tensor = IMAGE_TRANSFORM(paired_img)
            
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
        
        label_tensor = torch.tensor(label, dtype=torch.float32)

        doc1_modalities = (orig_tokens, orig_img_tensor)
        doc2_modalities = (paired_tokens, paired_img_tensor)
        
        return doc1_modalities, doc2_modalities, label_tensor


def collate_fn(batch):
    """Custom collate function for batching."""
    doc1_list, doc2_list, label_list = zip(*batch)
    
    doc1_texts, doc1_imgs = zip(*doc1_list)
    doc2_texts, doc2_imgs = zip(*doc2_list)
    
    doc1_imgs_batch = torch.stack(doc1_imgs)
    doc2_imgs_batch = torch.stack(doc2_imgs)
    
    doc1_text_batch = {
        key: torch.cat([d[key] for d in doc1_texts], dim=0)
        for key in doc1_texts[0].keys()
    }
    doc2_text_batch = {
        key: torch.cat([d[key] for d in doc2_texts], dim=0)
        for key in doc2_texts[0].keys()
    }
    
    labels_batch = torch.stack(label_list)
    
    return (doc1_text_batch, doc1_imgs_batch), (doc2_text_batch, doc2_imgs_batch), labels_batch


def get_dataloader(is_train: bool = True, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create DataLoader for training or testing."""
    data_df = pd.read_csv(DATASET_CSV_PATH)
    
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
        num_workers=0
    )


def get_full_dataloader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create DataLoader for entire dataset (evaluation)."""
    data_df = pd.read_csv(DATASET_CSV_PATH)
    dataset = ForensicsDataset(data_df)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )