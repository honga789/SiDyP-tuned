'''
Pre-process the dataset
'''

import random
import torch
import numpy as np
import pandas as pd

from math import inf
from scipy import stats
from utils import random_label_assign
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

'''Synthetic Noises: SN, ASN, IDN'''
def corrupt_dataset_SN(args, data):
    new_data = data.detach().clone()
    noise_ratio = args.noise_ratio * args.num_classes / (args.num_classes - 1)
    for i in range(len(new_data)):
        if random.random() > noise_ratio:
            continue
        else:
            new_data[i] = torch.randint(low=0, high=args.num_classes, size=(1, ))
    return new_data 

def corrupt_dataset_ASN(args, data):
    new_data = data.detach().clone()
    for i in range(len(new_data)):
        if random.random() > args.noise_ratio:
            continue
        else:
            new_data[i] = (new_data[i] + 1) % args.num_classes
    return new_data

def corrupt_dataset_IDN(args, inputs, labels):
    flip_distribution = stats.truncnorm((0-args.noise_ratio)/0.1, (1-args.noise_ratio)/0.1, loc=args.noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(labels))
    W = torch.randn(args.num_classes, inputs.shape[-1], args.num_classes).float()
    new_label = labels.detach().clone()
    for i in range(len(new_label)):
        p = inputs[i].float().view(1,-1).mm(W[labels[i].long()].squeeze(0)).squeeze(0)
        p[labels[i]] = -inf
        p = flip_rate[i] * torch.softmax(p, dim=0)
        p[labels[i]] += 1 - flip_rate[i]
        new_label[i] = torch.multinomial(p,1)
    return new_label 


def load_dataset(args):
    # Đọc train CSV
    train_df = pd.read_csv(args.train_csv_path)
    # Đọc feather nhãn nhiễu
    train_feather = pd.read_feather(args.train_feather_path)
    # Đọc test CSV
    test_df = pd.read_csv(args.test_csv_path)

    # Lấy dữ liệu và nhãn sạch
    train_texts = train_df[args.train_data_column].values
    train_true_labels = train_df[args.train_label_column].values
    train_true_labels = torch.tensor(train_true_labels, dtype=torch.long, device=args.device)

    # Lấy nhãn nhiễu từ feather
    train_noisy_labels = train_feather['label'].values
    train_noisy_labels = torch.tensor(train_noisy_labels, dtype=torch.long, device=args.device)

    # Test set
    test_texts = test_df[args.test_data_column].values
    test_true_labels = test_df[args.test_label_column].values
    test_true_labels = torch.tensor(test_true_labels, dtype=torch.long, device=args.device)

    # Chia valid từ train
    train_idx, valid_idx = train_test_split(np.arange(len(train_texts)), test_size=0.2, random_state=42, shuffle=True)
    valid_texts = train_texts[valid_idx]
    valid_true_labels = train_true_labels[valid_idx]
    valid_noisy_labels = train_noisy_labels[valid_idx]
    train_texts = train_texts[train_idx]
    train_true_labels = train_true_labels[train_idx]
    train_noisy_labels = train_noisy_labels[train_idx]

    return train_texts, train_true_labels, train_noisy_labels, valid_texts, valid_true_labels, valid_noisy_labels, test_texts, test_true_labels


def create_dataset(args):
    train_input_sent, train_true_labels, train_noisy_labels, valid_input_sent, valid_true_labels, \
            valid_noisy_labels, test_input_sent, test_true_labels = load_dataset(args)
    
    if args.dataset == "20news":
        MAX_LEN = 150
    elif args.dataset == "chemprot":
        MAX_LEN = 512
    else:
        MAX_LEN = 512

    # Encode train/test text
    # ===========================
    tokenizer = BertTokenizer.from_pretrained(args.plc, do_lower_case=True)
    train_input_ids = []
    train_attention_masks = []
    for sent in train_input_sent:
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = MAX_LEN,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )
        train_input_ids.append(encoded_sent)


    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in train_input_ids:
        seq_mask = [float(i>0) for i in seq]
        train_attention_masks.append(seq_mask)

    train_inputs = torch.tensor(train_input_ids, device=args.device)
    train_masks = torch.tensor(train_attention_masks, device=args.device)

    valid_input_ids = []
    valid_attention_masks = []
    for sent in valid_input_sent:
        encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        valid_input_ids.append(encoded_sent)

    valid_input_ids = pad_sequences(valid_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in valid_input_ids:
        seq_mask = [float(i>0) for i in seq]
        valid_attention_masks.append(seq_mask)

    valid_inputs = torch.tensor(valid_input_ids, device=args.device)
    valid_masks = torch.tensor(valid_attention_masks, device=args.device)

    test_input_ids = []
    test_attention_masks = []
    for sent in test_input_sent:
        encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        test_input_ids.append(encoded_sent)

    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask)

    test_inputs = torch.tensor(test_input_ids, device=args.device)
    test_masks = torch.tensor(test_attention_masks, device=args.device)

    if args.noise_type == "synthetic" and args.syn_type == "IDN":
        del train_noisy_labels, valid_noisy_labels
        train_noisy_labels = corrupt_dataset_IDN(args, train_inputs.cpu(), train_true_labels)
        valid_noisy_labels = corrupt_dataset_IDN(args, valid_inputs.cpu(), valid_true_labels)

    embedding_model = SentenceTransformer(args.embed)
    train_embedding = embedding_model.encode(train_input_sent, convert_to_tensor=True)
    valid_embedding = embedding_model.encode(valid_input_sent, convert_to_tensor=True)
    test_embedding = embedding_model.encode(test_input_sent, convert_to_tensor=True)


    train_data = TensorDataset(train_inputs, train_masks, train_true_labels, train_noisy_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = TensorDataset(valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_true_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    return train_data, train_sampler, train_dataloader, train_embedding, valid_data, valid_sampler, valid_dataloader, valid_embedding, test_data, test_sampler, test_dataloader, test_embedding
        