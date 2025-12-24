import torch
from torch.utils.data import DataLoader
from dataset import TripletImageDataset
from loss import BatchHardTripletLoss

def train(model, optimizer, train_ds, val_ds, epochs, device):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    criterion = BatchHardTripletLoss()
    best = float("inf")

    for e in range(epochs):
        model.train()
        tr = 0

        for a,p,n in train_loader:
            a,p,n = a.to(device),p.to(device),n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            tr += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for a,p,n in val_loader:
                a,p,n = a.to(device),p.to(device),n.to(device)
                vl += criterion(model(a), model(p), model(n)).item()

        if vl < best:
            best = vl
            torch.save(model.state_dict(), "best_model.pth")

        print(e, tr/len(train_loader), vl/len(val_loader))
