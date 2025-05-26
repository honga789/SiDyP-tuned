import time
import torch
import torch.nn.functional as F 
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class KNN_prior_dynamic:
    def __init__(self, args, data_embedding, noisy_labels, true_labels, noisy_markers):
        self.args = args
        
        self.y_hat = noisy_labels
        self.y = true_labels
        self.noisy_markers = noisy_markers
        self.data_embedding = data_embedding

    
    def get_dynamic_prior(self, k=10, weighted=True):
        knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', metric="l2")

        # get clean data point's embeddings and corresponding labels
        clean_embed = self.data_embedding[self.noisy_markers==0, :]
        clean_label = self.y_hat[self.noisy_markers==0]
        
        clean_embed = clean_embed.cpu().detach().numpy()
        clean_label = clean_label.cpu().detach().numpy()

        # fit knn classifier to clean embedding
        knn_classifier.fit(clean_embed, clean_label)

        # predict class of entire training dataset
        all_train_embed = self.data_embedding.cpu().detach().numpy()

        proba = knn_classifier.predict_proba(all_train_embed)

        proba = torch.tensor(proba)
        all_classes = set(range(self.args.num_classes))
        current_classes = set(clean_label.tolist())
        missing_classes = all_classes - current_classes
        if missing_classes:
            sorted_classes = sorted(list(missing_classes))
            new_proba_size = proba.size(1) + len(sorted_classes)
            new_proba = torch.zeros((proba.size(0), new_proba_size), dtype=proba.dtype)

            origin_pos = 0
            new_pos = 0
            for missing_class in sorted_classes:
                while new_pos < missing_class:
                    new_proba[:, new_pos] = proba[:, origin_pos]
                    origin_pos += 1
                    new_pos += 1
                
                if new_pos == missing_class:
                    new_pos += 1
                
            while new_pos <= proba.size(1):
                new_proba[:, new_pos] = proba[:, origin_pos]
                origin_pos += 1
                new_pos += 1
                

            proba = new_proba.clone()
        # get the potential neighbor labels with probability > 0
        neighbor = []
        for i in range(proba.size(0)):
            neighbor.append(torch.where(proba[i, :] > 0.0)[0].cpu().detach().numpy().tolist())
        
        neighbor = [torch.tensor(labels) for labels in neighbor]

        # get the max probability and its corresponding index
        max_proba, max_idx = torch.max(proba, dim=1)

        # get the marker for uncertain label
        uncertain_marker = max_proba < self.args.certain_threshold

        # convert certain data point's probability into one hot
        certain_proba = F.one_hot(max_idx[uncertain_marker==False], num_classes=self.args.num_classes).to(dtype=torch.float64)

        # change the original proba for certain data point
        proba[uncertain_marker==False] = certain_proba

        # filter out certain/determined labels
        uncertain_proba = proba[uncertain_marker==True]

        # get the max 2 probs and idx in uncertain sets
        top2_proba, top2_idx = torch.topk(uncertain_proba, 2)
        
        # check the sum of max 2 probs if it is over dominant threshold
        dominant_marker = torch.sum(top2_proba, dim=1) >= self.args.dominant_threshold

        # eliminate other trival labels if the sum is over dominant threshold
        dominant_idx = top2_idx[dominant_marker == True]

        # normalize the proability to make sure they add up to 1
        dominant_proba = top2_proba[dominant_marker==True] / torch.sum(top2_proba[dominant_marker==True], dim=1, keepdim=True)

        # initialize need-to-update uncertain probability (which is dominant above)
        update_uncertain_proba = torch.zeros_like(uncertain_proba[dominant_marker==True])
        row_idx = torch.arange(dominant_idx.size()[0])

        # broadcast to update dominant probability
        update_uncertain_proba[row_idx, dominant_idx[:, 0]] = dominant_proba[row_idx, 0]
        update_uncertain_proba[row_idx, dominant_idx[:, 1]] = dominant_proba[row_idx, 1]

        # update in original probability
        aux_uncertain_proba = proba[uncertain_marker==True]
        aux_uncertain_proba[dominant_marker==True] = update_uncertain_proba
        proba[uncertain_marker==True] = aux_uncertain_proba
        

        # get neighbor labels
        neighbor = []
        for i in range(proba.size()[0]):
            neighbor.append(torch.where(proba[i, :]>0.0)[0])

        # pad with -1 to maintain same size
        neighbor = pad_sequence(neighbor, batch_first=True, padding_value=-1)

        for idx,i in enumerate(neighbor):
            if torch.all(i == -1):
                embedding = all_train_embed[idx][np.newaxis, :]
                pred_y = knn_classifier.predict(embedding)[0]
                neighbor[idx][0] = pred_y
                uncertain_marker[idx] = False
                proba[idx][0] = 1.0

        return neighbor, proba, uncertain_marker, self.y