import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

class ImageTamperer:
    def __init__(self, orig_root="dataset/originals", tamp_root="dataset/tampered"):
        self.orig_root = orig_root
        self.tamp_root = tamp_root

        for s in ["train","val","test"]:
            os.makedirs(os.path.join(tamp_root, s), exist_ok=True)

    def copy_move(self, img):
        w,h = img.size
        pw,ph = random.randint(60,160), random.randint(60,160)
        x1,y1 = random.randint(0,w-pw), random.randint(0,h-ph)
        x2,y2 = random.randint(0,w-pw), random.randint(0,h-ph)
        patch = img.crop((x1,y1,x1+pw,y1+ph))
        img.paste(patch,(x2,y2))
        return img

    def noise_patch(self, img):
        arr = np.array(img).astype(np.float32)
        h,w,_ = arr.shape
        rw,rh = random.randint(w//4,w//2), random.randint(h//4,h//2)
        x,y = random.randint(0,w-rw), random.randint(0,h-rh)
        noise = np.random.normal(0,50,(rh,rw,3))
        arr[y:y+rh,x:x+rw] = np.clip(arr[y:y+rh,x:x+rw]+noise,0,255)
        return Image.fromarray(arr.astype(np.uint8))

    def splicing(self, img, donors):
        donor = Image.open(random.choice(donors)).convert("RGB")
        pw,ph = random.randint(60,160), random.randint(60,160)
        x,y = random.randint(0,donor.width-pw), random.randint(0,donor.height-ph)
        patch = donor.crop((x,y,x+pw,y+ph))
        px,py = random.randint(0,img.width-pw), random.randint(0,img.height-ph)
        img.paste(patch,(px,py))
        return img

    def blur_region(self, img):
        w,h = img.size
        pw,ph = random.randint(80,200), random.randint(80,200)
        x,y = random.randint(0,w-pw), random.randint(0,h-ph)
        region = img.crop((x,y,x+pw,y+ph)).filter(ImageFilter.GaussianBlur(6))
        img.paste(region,(x,y))
        return img

    def color_shift(self, img):
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.2,1.8))

    def create(self, per_image=4):
        for split in ["train","val","test"]:
            orig_dir = os.path.join(self.orig_root, split)
            out_dir = os.path.join(self.tamp_root, split)

            files = [os.path.join(orig_dir,f) for f in os.listdir(orig_dir)]
            methods = ["copy","noise","splice","blur","color"]

            for path in files:
                base = os.path.splitext(os.path.basename(path))[0]
                img = Image.open(path).convert("RGB")

                for i in range(per_image):
                    m = random.choice(methods)
                    if m == "copy":
                        out = self.copy_move(img.copy())
                    elif m == "noise":
                        out = self.noise_patch(img.copy())
                    elif m == "splice":
                        out = self.splicing(img.copy(), files)
                    elif m == "blur":
                        out = self.blur_region(img.copy())
                    else:
                        out = self.color_shift(img.copy())

                    name = f"{base}_tamp{i}_{m}.jpg"
                    out.save(os.path.join(out_dir, name))
