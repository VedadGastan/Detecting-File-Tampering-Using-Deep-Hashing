import os
import time
import random
import requests
import hashlib

class ImageScraper:
    def __init__(self, root="dataset/originals"):
        self.root = root
        for s in ["train","val","test"]:
            os.makedirs(os.path.join(root, s), exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DeepHashDatasetBuilder/1.0"})

        self.sleep = 0.3

    def _hash(self, b):
        return hashlib.md5(b).hexdigest()

    def _count_existing(self, split):
        return len([f for f in os.listdir(os.path.join(self.root, split)) if f.endswith(".jpg")])

    def download(self, total=3000, split=(0.7,0.15,0.15), w=800, h=600):
        targets = {
            "train": int(total * split[0]),
            "val": int(total * split[1]),
            "test": total - int(total * split[0]) - int(total * split[1])
        }

        seen = set()

        for split_name, target in targets.items():
            saved = self._count_existing(split_name)
            out_dir = os.path.join(self.root, split_name)

            while saved < target:
                seed = random.randint(1, 100_000_000)
                url = f"https://picsum.photos/seed/{seed}/{w}/{h}"

                try:
                    r = self.session.get(url, timeout=8)
                    if "image" not in r.headers.get("Content-Type",""):
                        raise RuntimeError

                    hsh = self._hash(r.content)
                    if hsh in seen:
                        raise RuntimeError
                    seen.add(hsh)

                    path = os.path.join(out_dir, f"img_{saved:05d}.jpg")
                    with open(path, "wb") as f:
                        f.write(r.content)

                    saved += 1
                    self.sleep = max(0.2, self.sleep * 0.97)

                except:
                    self.sleep = min(2.0, self.sleep * 1.15)

                time.sleep(self.sleep)
