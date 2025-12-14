import torch
import torch.nn as nn
import fitz
import numpy as np
import os
import json
import math
from pathlib import Path

class Config:
    INPUT_DIM = 22
    HASH_BITS = 128
    MODEL_PATH = "forensic_hash_model.pth"
    HASH_DB_PATH = "hash_registry.json"
    TRAIN_DIR = "train_data"
    TEST_DIR = "test_data"
    HAMMING_THRESHOLD = 4

class PDFFeaturizer:
    FEATURE_NAMES = [
        "page_count", "file_size_log", "stream_count", "metadata_len",
        "image_count", "font_count", "text_len", "entropy",
        "canvas_width", "canvas_height", "has_js", "is_encrypted",
        "xref_len", "obj_count", "info_dict_len", "pdf_version",
        "attachment_count", "link_count", "annot_count", "text_annot_count",
        "has_ocg", "has_acroform"
    ]

    @staticmethod
    def safe_extract(func, default=0):
        try:
            return func()
        except Exception:
            return default

    @staticmethod
    def get_file_size(path):
        try:
            return os.path.getsize(path)
        except:
            return 0

    @staticmethod
    def process(doc_or_path):
        doc = None
        close_doc = False
        
        try:
            if isinstance(doc_or_path, str):
                if not Path(doc_or_path).exists():
                    return torch.zeros(Config.INPUT_DIM)
                try:
                    doc = fitz.open(doc_or_path)
                    close_doc = True
                except Exception as e:
                    print(f"Error opening {doc_or_path}: {e}")
                    return torch.zeros(Config.INPUT_DIM)
            else:
                doc = doc_or_path
            
            features = []
            
            features.append(doc.page_count)
            
            if isinstance(doc_or_path, str):
                file_size = PDFFeaturizer.get_file_size(doc_or_path)
            else:
                file_size = len(doc.write())
            features.append(math.log1p(file_size) if file_size > 0 else 0)
            
            buffer = doc.write()
            features.append(buffer.count(b"stream"))
            
            metadata = doc.metadata
            features.append(len(str(metadata)))
            
            if doc.page_count > 0:
                page = doc[0]
                images = page.get_images()
                features.append(len(images))
                
                fonts = list(page.get_fonts())
                features.append(len(fonts))
                
                text = page.get_text()
                features.append(len(text))
                
                if text and len(text) > 0:
                    non_alnum = sum(1 for c in text if not c.isalnum())
                    features.append(non_alnum / len(text))
                else:
                    features.append(0.0)
                
                rect = page.rect
                features.append(rect.width)
                features.append(rect.height)
            else:
                features.extend([0, 0, 0, 0.0, 0.0, 0.0])
            
            features.append(1.0 if doc.is_pdf and "javascript" in str(metadata).lower() else 0.0)
            features.append(1.0 if doc.is_encrypted else 0.0)
            
            try:
                xref_length = doc.xref_length()
                features.append(xref_length)
            except:
                features.append(0)
            
            features.append(buffer.count(b" obj "))
            
            try:
                info_dict = doc.pdf_catalog()
                features.append(len(str(info_dict)) if info_dict else 0)
            except:
                features.append(0)
            
            try:
                version = float(doc.pdf_version)
                features.append(version)
            except:
                features.append(1.4)
            
            try:
                embfile_count = len(doc.embfile_names())
                features.append(embfile_count)
            except:
                features.append(0)
            
            if doc.page_count > 0:
                page = doc[0]
                links = page.get_links()
                features.append(len(links))
                
                annots = list(page.annots())
                features.append(len(annots))
                
                text_annots = [a for a in annots if hasattr(a, 'type') and a.type[1] == "Text"]
                features.append(len(text_annots))
            else:
                features.extend([0, 0, 0])
            
            has_ocg = 0
            try:
                root = doc.pdf_catalog()
                if root and "OCProperties" in root:
                    has_ocg = 1
            except:
                pass
            features.append(has_ocg)
            
            has_acroform = 0
            try:
                root = doc.pdf_catalog()
                if root and "AcroForm" in root:
                    has_acroform = 1
            except:
                pass
            features.append(has_acroform)
            
            return torch.FloatTensor(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return torch.zeros(Config.INPUT_DIM)
        finally:
            if close_doc and doc:
                doc.close()

class DeepHashModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(Config.INPUT_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, Config.HASH_BITS),
            nn.Tanh()
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.bn_input(x)
        return self.encoder(x)
    
    def get_hex_hash(self, x):
        self.eval()
        with torch.no_grad():
            h = self.forward(x)
            bits = (h > 0).int().cpu().numpy().flatten()
            if len(bits) == 0:
                return "0" * (Config.HASH_BITS // 4)
            hex_str = hex(int("".join(map(str, bits)), 2))[2:].upper()
            return hex_str.zfill(Config.HASH_BITS // 4)

class HashDatabase:
    def __init__(self):
        self.path = Config.HASH_DB_PATH
        self.db = self._load()
    
    def _load(self):
        if Path(self.path).exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def add(self, name, hash_val, features):
        if torch.is_tensor(features):
            features = features.tolist()
        self.db[name] = {
            'hash': hash_val,
            'features': features
        }
        self.save()
    
    def find_match(self, query_hash):
        if not self.db:
            return None, float('inf')
        
        try:
            q_int = int(query_hash, 16)
            q_bits = bin(q_int)[2:].zfill(Config.HASH_BITS)
        except:
            return None, float('inf')
        
        min_dist = float('inf')
        match_name = None
        
        for name, entry in self.db.items():
            try:
                s_int = int(entry['hash'], 16)
                s_bits = bin(s_int)[2:].zfill(Config.HASH_BITS)
                dist = sum(1 for b1, b2 in zip(q_bits, s_bits) if b1 != b2)
                if dist < min_dist:
                    min_dist = dist
                    match_name = name
            except:
                continue
        
        return match_name, min_dist
    
    def clear(self):
        self.db = {}
        self.save()