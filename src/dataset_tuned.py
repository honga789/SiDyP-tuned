'''
Pre-process the dataset
'''

import random
import torch
import numpy as np
import pandas as pd
import open_clip

from PIL import Image
from pathlib import Path
from math import inf
from scipy import stats
from utils import random_label_assign
from transformers import BertTokenizer
# from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import train_test_split

# Drop-in replacement for keras.preprocessing.sequence.pad_sequences
def pad_sequences(sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0):

    # ---- dtype normalization (accept case-insensitive & aliases) ----
    if isinstance(dtype, str):
        key = dtype.lower()
        dtype_map = {
            "int32": np.int32,
            "int64": np.int64,
            "long":  np.int64,    # alias
            "float32": np.float32,
            "float64": np.float64,
            "object": np.object_, # Keras cho phép dùng object
        }
        np_dtype = np.dtype(dtype_map.get(key, key))
    else:
        np_dtype = np.dtype(dtype)

    if padding not in {"pre", "post"}:
        raise ValueError(f"padding must be 'pre' or 'post' (got {padding})")
    if truncating not in {"pre", "post"}:
        raise ValueError(f"truncating must be 'pre' or 'post' (got {truncating})")

    # ---- sequences must be iterable of iterables ----
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    sequences = list(sequences)
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    got_sample_shape = False

    for s in sequences:
        try:
            l = len(s)
        except TypeError as e:
            raise ValueError("`sequences` must be a list of iterables. Found non-iterable element.") from e
        lengths.append(l)
        if (not got_sample_shape) and l:
            arr0 = np.asarray(s)
            if arr0.ndim > 1:
                sample_shape = arr0.shape[1:]
            got_sample_shape = True

    if maxlen is None:
        maxlen = max(lengths) if lengths else 0
    if maxlen < 0:
        raise ValueError("`maxlen` must be >= 0.")

    # Trường hợp maxlen == 0: trả về mảng rỗng đúng shape
    if maxlen == 0:
        return np.full((num_samples, 0) + sample_shape, value, dtype=np_dtype)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=np_dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        arr = np.asarray(s)

        # chiều timesteps
        L = len(arr)
        trunc_len = min(L, maxlen)
        if trunc_len == 0:
            continue

        if truncating == "pre":
            start = L - trunc_len
            trunc = arr[start:]
        else:  # "post"
            trunc = arr[:trunc_len]

        # Gán theo padding
        if padding == "post":
            x[idx, :trunc_len] = trunc
        else:  # "pre"
            x[idx, maxlen - trunc_len:] = trunc

    return x

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
    """
    Đọc dataset tuỳ theo args.data_type:
    - 'text': giữ nguyên hành vi cũ (đọc cột văn bản)
    - 'image': đọc cột tên file ảnh và ghép với --train_image_path / --test_image_path
    Feather train vẫn chứa nhãn nhiễu ở cột 'label'.
    """

    # Đọc train/test CSV và feather (nhãn nhiễu cho train)
    train_df = pd.read_csv(args.train_csv_path)
    train_feather = pd.read_feather(args.train_feather_path)
    test_df = pd.read_csv(args.test_csv_path)

    is_image = getattr(args, "data_type", "text") == "image"

    # --- Train inputs ---
    if is_image:
        if args.train_image_path is None or args.test_image_path is None:
            raise ValueError("For data_type='image', please set both --train_image_path and --test_image_path.")
        base_train = Path(args.train_image_path)
        train_inputs = train_df[args.train_data_column].astype(str).apply(lambda x: str(base_train / x)).values
    else:
        train_inputs = train_df[args.train_data_column].values  # text

    # Nhãn sạch (train) và nhãn nhiễu từ feather (train)
    train_true_labels = torch.tensor(train_df[args.train_label_column].values, dtype=torch.long, device=args.device)
    train_noisy_labels = torch.tensor(train_feather["label"].values, dtype=torch.long, device=args.device)

    # --- Test inputs ---
    if is_image:
        base_test = Path(args.test_image_path)
        test_inputs = test_df[args.test_data_column].astype(str).apply(lambda x: str(base_test / x)).values
    else:
        test_inputs = test_df[args.test_data_column].values  # text

    # Nhãn sạch (test)
    test_true_labels = torch.tensor(test_df[args.test_label_column].values, dtype=torch.long, device=args.device)

    # Kích thước gốc của tập train
    orig_train_size = len(train_inputs)

    # Chia valid từ train (giữ nguyên logic cũ)
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_inputs)), test_size=0.2, random_state=42, shuffle=True
    )

    # Tách train/valid cho inputs và labels
    valid_inputs = train_inputs[valid_idx]
    valid_true_labels = train_true_labels[valid_idx]
    valid_noisy_labels = train_noisy_labels[valid_idx]

    train_inputs = train_inputs[train_idx]
    train_true_labels = train_true_labels[train_idx]
    train_noisy_labels = train_noisy_labels[train_idx]

    # Giữ nguyên định dạng trả về để create_dataset dùng được
    return (
        train_inputs, train_true_labels, train_noisy_labels,
        valid_inputs, valid_true_labels, valid_noisy_labels,
        test_inputs, test_true_labels,
        orig_train_size, train_idx, valid_idx
    )

def create_dataset(args):
    """
    Mở rộng để hỗ trợ ảnh bằng open_clip (ViT-B-16, weights 'datacomp_xl_s13b_b90k')
    và GIỮ NGUYÊN hành vi cũ cho text.

    Trả về (không đổi chữ ký):
        train_data, train_sampler, train_dataloader, train_embedding,
        valid_data, valid_sampler, valid_dataloader, valid_embedding,
        test_data,  test_sampler,  test_dataloader,  test_embedding,
        orig_train_size, train_idx, valid_idx
    """

    is_image = getattr(args, "data_type", "text") == "image"

    if is_image:
        # ================= IMAGE PIPELINE (open_clip) =================
        train_inputs_raw, train_true_labels, train_noisy_labels, \
        valid_inputs_raw, valid_true_labels, valid_noisy_labels, \
        test_inputs_raw,  test_true_labels, \
        orig_train_size, train_idx, valid_idx = load_dataset(args)

        device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 1) Model + preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="datacomp_xl_s13b_b90k"
        )
        model = model.to(device).eval()

        # 2) Dataset đọc ảnh on-the-fly; giữ chỗ "mask" để không phá vỡ tuple (input, mask, label,...)
        class _ImageDataset(torch.utils.data.Dataset):
            def __init__(self, paths, true_labels, noisy_labels=None, preprocess=None):
                self.paths = list(paths)
                self.true_labels = true_labels
                self.noisy_labels = noisy_labels
                self.preprocess = preprocess

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                path = str(self.paths[idx])
                img = Image.open(path).convert("RGB")
                px = self.preprocess(img)  # [3, H, W]
                true = torch.as_tensor(int(self.true_labels[idx]), dtype=torch.long)
                mask = torch.zeros(1, dtype=torch.long)  # placeholder cho ảnh
                if self.noisy_labels is not None:
                    noisy = torch.as_tensor(int(self.noisy_labels[idx]), dtype=torch.long)
                    return px, mask, true, noisy
                else:
                    return px, mask, true

        train_data = _ImageDataset(train_inputs_raw, train_true_labels, train_noisy_labels, preprocess)
        valid_data = _ImageDataset(valid_inputs_raw, valid_true_labels, valid_noisy_labels, preprocess)
        test_data  = _ImageDataset(test_inputs_raw,  test_true_labels,  None,          preprocess)

        # 3) Sampler + DataLoader
        train_sampler = SequentialSampler(train_data)
        valid_sampler = SequentialSampler(valid_data)
        test_sampler  = SequentialSampler(test_data)

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size,
            num_workers=getattr(args, "num_workers", 0), pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size,
            num_workers=getattr(args, "num_workers", 0), pin_memory=True
        )
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size,
            num_workers=getattr(args, "num_workers", 0), pin_memory=True
        )

        # 4) Tính EMBEDDING ảnh bằng open_clip
        @torch.no_grad()
        def _encode_image_batches(dataloader):
            feats = []
            for batch in dataloader:
                pixel_values = batch[0].to(device, non_blocking=True)
                out = model.encode_image(pixel_values)     # [B, D]
                feats.append(out.detach().cpu().float())   # để CPU float32
            return torch.cat(feats, dim=0)

        train_embedding = _encode_image_batches(train_dataloader)
        valid_embedding = _encode_image_batches(valid_dataloader)
        test_embedding  = _encode_image_batches(test_dataloader)

        return (
            train_data, train_sampler, train_dataloader, train_embedding,
            valid_data, valid_sampler, valid_dataloader, valid_embedding,
            test_data,  test_sampler,  test_dataloader,  test_embedding,
            orig_train_size, train_idx, valid_idx
        )

    # ================= TEXT PIPELINE =================
    train_input_sent, train_true_labels, train_noisy_labels, \
    valid_input_sent, valid_true_labels, valid_noisy_labels, \
    test_input_sent, test_true_labels, \
    orig_train_size, train_idx, valid_idx = load_dataset(args)
    
    if args.dataset == "20news":
        MAX_LEN = 150
    elif args.dataset == "chemprot":
        MAX_LEN = 512
    else:
        MAX_LEN = 128

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
                            truncation=True,
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
                                truncation=True,
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
                                truncation=True,
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

    if args.embed == "bert-base-uncased":
        bert = models.Transformer('bert-base-uncased')
        pool = models.Pooling(bert.get_word_embedding_dimension(),
                            pooling_mode_mean_tokens=True)
        embedding_model = SentenceTransformer(modules=[bert, pool])
    else:
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

    return (
        train_data, train_sampler, train_dataloader, train_embedding,
        valid_data, valid_sampler, valid_dataloader, valid_embedding,
        test_data, test_sampler, test_dataloader, test_embedding,
        orig_train_size, train_idx, valid_idx
    )
