"""
Enhanced Dataset Loader with Structural Features
Loads: Text + Image + PDF Structure
"""

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from typing import Tuple, Dict
from config import IMAGE_SIZE, MAX_SEQ_LENGTH, BATCH_SIZE, TRAIN_TEST_SPLIT
import numpy as np
import os

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')

PREPROCESSED_CSV_PATH = "data/preprocessed/dataset_preprocessed.csv"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ForensicsDataset(Dataset):
    """
    Enhanced dataset for paired PDF documents with 3 modalities:
    Text, Image, and Structural Features
    """
    
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df
        print(f"Loaded {len(self.data_df)} pre-processed samples")
        
    def __len__(self) -> int:
        return len(self.data_df)

    def _load_modalities(self, txt_path: str, img_path: str, struct_path: str) -> Tuple:
        """Loads text, image, and structural features from pre-processed files."""
        
        # Load text
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Warning: Text file not found {txt_path}")
            text = ""
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
        
        # NEW: Load structural features
        try:
            structural_features = np.load(struct_path)
            structural_tensor = torch.from_numpy(structural_features).float()
        except FileNotFoundError:
            print(f"Warning: Structural file not found {struct_path}")
            structural_tensor = torch.zeros(40, dtype=torch.float32)
        
        return text, image, structural_tensor

    def __getitem__(self, idx: int) -> Tuple:
        row = self.data_df.iloc[idx]
        
        # Load all modalities for both documents
        orig_text, orig_img, orig_struct = self._load_modalities(
            row['Original_Text_Path'], 
            row['Original_Image_Path'],
            row['Original_Structural_Path']
        )
        
        paired_text, paired_img, paired_struct = self._load_modalities(
            row['Paired_Text_Path'], 
            row['Paired_Image_Path'],
            row['Paired_Structural_Path']
        )

        # Transform images
        orig_img_tensor = IMAGE_TRANSFORM(orig_img)
        paired_img_tensor = IMAGE_TRANSFORM(paired_img)
        
        # Tokenize text
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
        
        label_tensor = torch.tensor(row['Label'], dtype=torch.float32)

        # Return all three modalities for each document
        doc1_modalities = (orig_tokens, orig_img_tensor, orig_struct)
        doc2_modalities = (paired_tokens, paired_img_tensor, paired_struct)
        
        return doc1_modalities, doc2_modalities, label_tensor


def collate_fn(batch):
    """Custom collate function for batching with structural features."""
    doc1_list, doc2_list, label_list = zip(*batch)
    
    # Unpack all three modalities
    doc1_texts, doc1_imgs, doc1_structs = zip(*doc1_list)
    doc2_texts, doc2_imgs, doc2_structs = zip(*doc2_list)
    
    # Stack images
    doc1_imgs_batch = torch.stack(doc1_imgs)
    doc2_imgs_batch = torch.stack(doc2_imgs)
    
    # Stack structural features
    doc1_structs_batch = torch.stack(doc1_structs)
    doc2_structs_batch = torch.stack(doc2_structs)
    
    # Batch text tokens
    doc1_text_batch = {
        key: torch.cat([d[key] for d in doc1_texts], dim=0)
        for key in doc1_texts[0].keys()
    }
    doc2_text_batch = {
        key: torch.cat([d[key] for d in doc2_texts], dim=0)
        for key in doc2_texts[0].keys()
    }
    
    labels_batch = torch.stack(label_list)
    
    return (
        (doc1_text_batch, doc1_imgs_batch, doc1_structs_batch), 
        (doc2_text_batch, doc2_imgs_batch, doc2_structs_batch), 
        labels_batch
    )


def get_dataloader(is_train: bool = True, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create DataLoader for training or testing."""
    
    if not os.path.exists(PREPROCESSED_CSV_PATH):
        raise FileNotFoundError(
            f"{PREPROCESSED_CSV_PATH} not found.\n"
            "Please run 'python preprocess_data.py' before training."
        )
        
    data_df = pd.read_csv(PREPROCESSED_CSV_PATH)
    
    train_size = int(len(data_df) * TRAIN_TEST_SPLIT)
    
    if is_train:
        subset_df = data_df.iloc[:train_size]
        shuffle = True
    else:
        subset_df = data_df.iloc[train_size:]
        shuffle = False
    
    dataset = ForensicsDataset(subset_df)
    
    num_workers = 4 if os.name == 'posix' else 0  # Reduced for stability
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def get_full_dataloader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create DataLoader for entire dataset (evaluation)."""
    
    if not os.path.exists(PREPROCESSED_CSV_PATH):
        raise FileNotFoundError(
            f"{PREPROCESSED_CSV_PATH} not found.\n"
            "Please run 'python preprocess_data.py' before evaluation."
        )
        
    data_df = pd.read_csv(PREPROCESSED_CSV_PATH)
    dataset = ForensicsDataset(data_df)
    
    num_workers = 4 if os.name == 'posix' else 0
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )