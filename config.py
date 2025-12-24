import torch
from torchvision import transforms

SEED = 42
HASH_BITS = 128
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_ORIG = "dataset/originals/train"
VAL_ORIG = "dataset/originals/val"
TEST_ORIG = "dataset/originals/test"

TRAIN_TAMP = "dataset/tampered/train"
VAL_TAMP = "dataset/tampered/val"
TEST_TAMP = "dataset/tampered/test"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
