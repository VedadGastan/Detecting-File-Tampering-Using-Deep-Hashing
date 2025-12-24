import os
import random
from torch.utils.data import Dataset
from PIL import Image

class TripletImageDataset(Dataset):
    def __init__(self, original_dir, tampered_dir, transform):
        self.transform = transform

        self.originals = {}
        for f in os.listdir(original_dir):
            key = os.path.splitext(f)[0]
            self.originals[key] = os.path.join(original_dir, f)

        self.tampered = {}
        for f in os.listdir(tampered_dir):
            key = f.split("_tamp")[0]
            self.tampered.setdefault(key, []).append(os.path.join(tampered_dir, f))

        self.keys = list(set(self.originals) & set(self.tampered))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]

        a = Image.open(self.originals[k]).convert("RGB")
        p = Image.open(self.originals[k]).convert("RGB")
        n = Image.open(random.choice(self.tampered[k])).convert("RGB")

        return self.transform(a), self.transform(p), self.transform(n)
