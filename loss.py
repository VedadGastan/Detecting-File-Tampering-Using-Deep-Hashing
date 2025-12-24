import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.4, alpha=0.01):
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, a, p, n):
        emb = torch.cat([a, p, n])
        labels = torch.arange(len(a), device=a.device).repeat(3)

        dist = torch.cdist(emb, emb)
        loss = []

        for i in range(len(emb)):
            pos = dist[i][labels == labels[i]]
            neg = dist[i][labels != labels[i]]
            if len(pos) > 1 and len(neg) > 0:
                loss.append(torch.clamp(pos.max() - neg.min() + self.margin, min=0))

        loss = torch.stack(loss).mean()
        reg = ((emb.abs() - 1) ** 2).mean()

        return loss + self.alpha * reg
