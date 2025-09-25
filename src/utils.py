import torch
import numpy as np
import random

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def random_label_assign(args, noisy_label):
    if torch.any(noisy_label == -1):
        no_answer_mask = noisy_label == -1
        random_labels = torch.randint(0, args.num_classes, size=(no_answer_mask.sum().item(),), dtype=torch.long, device=args.device)
        noisy_label[no_answer_mask] = random_labels
    return noisy_label


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

def euclidean_dist(args, train_embeds, train_labels, train_labels2=None):
    mask = (train_labels != -1)
    train_embeds = train_embeds[mask]
    train_labels = train_labels[mask]

    if train_embeds.numel() == 0:
        raise ValueError("No training embeddings after filtering label = -1.")

    C = int(args.num_classes)
    D = train_embeds.shape[-1]
    device = train_embeds.device
    dtype = train_embeds.dtype

    centroids = torch.empty((C, D), device=device, dtype=dtype)
    present = torch.zeros(C, dtype=torch.bool, device=device)

    for c in range(C):
        cls_mask = (train_labels == c)
        if cls_mask.any():
            centroids[c] = train_embeds[cls_mask].mean(dim=0)
            present[c] = True
        else:
            centroids[c] = 0

    diff = train_embeds.unsqueeze(1) - centroids.unsqueeze(0)  # [N, C, D]
    dists = diff.pow(2).sum(dim=-1).sqrt()                     # [N, C]

    missing = ~present
    if missing.any():
        dists[:, missing] = float('inf')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dists

# def euclidean_dist(args, train_embeds, train_labels, train_labels2=None):
#     print(train_labels.shape)
#     train_embeds = train_embeds[train_labels!=-1]
#     train_labels = train_labels[train_labels!=-1]
#     l_max = int(torch.max(train_labels).item())
#     cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
#     for i in range(l_max+1):
#         cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
#     embeds1 = train_embeds.unsqueeze(1).repeat((1, cluster_centroids.shape[0], 1))
#     embeds2 = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1, 1))
#     dists = torch.sqrt(torch.sum((embeds1.to(embeds2.device) - embeds2) ** 2, -1)).to(embeds1.device)
#     print(dists.shape)
#     torch.cuda.empty_cache()
#     return dists