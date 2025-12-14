import torch
import torch.nn as nn
import fitz
import numpy as np
import os
import json

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
        "xref_len", "obj_count", "trailer_size", "pdf_version",
        "attachment_count", "link_count", "annot_count", "text_annot_count",
        "ocg_count", "js_stream_len"
    ]

    @staticmethod
    def process(doc_or_path):
        try:
            if isinstance(doc_or_path, str):
                if not os.path.exists(doc_or_path): return torch.zeros(Config.INPUT_DIM)
                doc = fitz.open(doc_or_path)
                raw_bytes = doc.tobytes()
            else:
                doc = doc_or_path
                raw_bytes = doc.tobytes()

            page = doc[0] if doc.page_count > 0 else None
            meta = str(doc.metadata)
            text = page.get_text() if page else ""
            
            features = [
                doc.page_count,
                np.log1p(len(raw_bytes)),
                raw_bytes.count(b"stream"),
                len(meta),
                len(page.get_images()) if page else 0,
                len(page.get_fonts()) if page else 0,
                len(text),
                sum(not c.isalnum() for c in text) / (len(text) + 1),
                page.rect.width if page else 0,
                page.rect.height if page else 0,
                1.0 if "JavaScript" in str(doc.get_xml_metadata()) else 0.0,
                1.0 if doc.is_encrypted else 0.0,
                doc.xref_length(),
                len(raw_bytes.split(b"obj")) - 1,
                len(doc.trailer_string()) if hasattr(doc, 'trailer_string') else 0,
                float(doc.version),
                doc.embfile_count(),
                len(page.get_links()) if page else 0,
                len(list(page.annots())) if page and page.annots() else 0,
                sum(1 for a in page.annots() if a.type[0] == 0) if page and page.annots() else 0,
                len(doc.get_ocgs()) if hasattr(doc, 'get_ocgs') else 0,
                len(doc.get_xml_metadata() or "")
            ]
            
            if isinstance(doc_or_path, str): doc.close()
            return torch.FloatTensor(features)
        except:
            return torch.zeros(Config.INPUT_DIM)

class DeepHashModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(Config.INPUT_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, Config.HASH_BITS)
        )
        
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = self.bn_input(x)
        return torch.tanh(self.encoder(x))
    
    def get_hex_hash(self, x):
        self.eval()
        with torch.no_grad():
            h = self.forward(x)
            bits = (h > 0).int().cpu().numpy().flatten()
            return hex(int("".join(map(str, bits)), 2))[2:].zfill(Config.HASH_BITS // 4).upper()

class HashDatabase:
    def __init__(self):
        self.path = Config.HASH_DB_PATH
        self.db = self._load()
    
    def _load(self):
        return json.load(open(self.path, 'r')) if os.path.exists(self.path) else {}
    
    def save(self):
        json.dump(self.db, open(self.path, 'w'), indent=2)
    
    def add(self, name, hash_val, features):
        self.db[name] = {'hash': hash_val, 'features': features.tolist()}
        self.save()

    def find_match(self, query_hash):
        if not self.db: return None, float('inf')
        q_bits = bin(int(query_hash, 16))[2:].zfill(Config.HASH_BITS)
        
        results = []
        for name, entry in self.db.items():
            s_bits = bin(int(entry['hash'], 16))[2:].zfill(Config.HASH_BITS)
            dist = sum(b1 != b2 for b1, b2 in zip(q_bits, s_bits))
            results.append((name, dist))
        
        return min(results, key=lambda x: x[1])