import torch
import torch.optim as optim
from config import *
from model import TripletHashNet
from dataset import TripletImageDataset
from train import train
from evaluate import evaluate
from utils import set_seed

def main():
    set_seed(SEED)

    model = TripletHashNet(HASH_BITS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_ds = TripletImageDataset(TRAIN_ORIG, TRAIN_TAMP, TRAIN_TRANSFORM)
    val_ds = TripletImageDataset(VAL_ORIG, VAL_TAMP, EVAL_TRANSFORM)

    #train(model, optimizer, train_ds, val_ds, EPOCHS, DEVICE)

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    results = evaluate(model, TEST_ORIG, TEST_TAMP, EVAL_TRANSFORM, DEVICE)
    print(results)

if __name__ == "__main__":
    main()
