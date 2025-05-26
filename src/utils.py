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
    print(train_labels.shape)
    train_embeds = train_embeds[train_labels!=-1]
    train_labels = train_labels[train_labels!=-1]
    l_max = int(torch.max(train_labels).item())
    cluster_centroids = torch.zeros((l_max+1, train_embeds.shape[-1]))
    for i in range(l_max+1):
        cluster_centroids[i] = torch.mean(train_embeds[train_labels==i], 0)
    embeds1 = train_embeds.unsqueeze(1).repeat((1, cluster_centroids.shape[0], 1))
    embeds2 = cluster_centroids.unsqueeze(0).repeat((train_embeds.shape[0], 1, 1))
    dists = torch.sqrt(torch.sum((embeds1.to(embeds2.device) - embeds2) ** 2, -1)).to(embeds1.device)
    print(dists.shape)
    torch.cuda.empty_cache()
    return dists