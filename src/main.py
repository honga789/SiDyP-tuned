import argparse
import torch
import random
from rich.traceback import install
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from dataset import create_dataset
from utils import set_seed
from plc_finetune import PLC_Trainer
from knn import KNN_prior_dynamic
from simplex_diff_trainer import Simplex_Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="numclaim", type=str, help="dataset:[semeval, numclaim, chemprot, trec, 20news]")
    parser.add_argument("--noise_type", default="llm", type=str, help="label noise type:[llm, realworld, synthetic]")
    parser.add_argument("--llm_type", default="llama3-70b", type=str, help="llm model:[llama3-70b, gpt4o, mixtral822, llama31-70b, llama31-405b]")
    parser.add_argument("--prompt_type", default="zeroshot", type=str, help="llm prompting method:[zeroshot, fewshot]")
    parser.add_argument("--syn_type", default="SN", type=str, help="synthetic noise type:[SN, ASN, IDN]")
    # plc
    parser.add_argument("--plc", default="bert-base-uncased", type=str, help="pretrain language classifier model")
    parser.add_argument("--embed", default="WhereIsAI/UAE-Large-V1", type=str, help="embedding model for knn classifier")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for validation.")
    parser.add_argument('--alpha_t', type=float, default=5)
    parser.add_argument("--plc_epochs", default=10, type=int, help="Number of epochs for PLC training.")
    parser.add_argument("--plc_lr", default=5e-5, type=float, help="The initial learning rate for PLC Adam.")
    parser.add_argument("--num_model", type=float, default=3, help='The number of model branches')
    # dynamic prior retrieval
    parser.add_argument("--noise_ratio", type=float, default=0.3, help='The ratio of noisy data to be poisoned.')
    parser.add_argument("--K", type=int, default=10, help="certain label retrieval threshold")
    parser.add_argument("--certain_threshold", type=float, default=0.9, help="certain label retrieval threshold")
    parser.add_argument("--dominant_threshold", type=float, default=0.8, help="dominant label retrieval threshold")
    # diffusion
    parser.add_argument("--diff_epochs", type=int, default=10, help="training epochs for diffusion model")
    parser.add_argument("--warmup_epochs", default=0.2, help="warmup_epochs", type=float)
    parser.add_argument("--diff_lr", default=1e-3, type=float, help="The initial learning rate for diffusion Adam.")
    parser.add_argument("--train_timesteps", default=500, help="Number of training timesteps for diffusion model", type=int)
    parser.add_argument("--infer_timesteps", default=10, help="Number of inference timesteps for diffusion model", type=int)
    parser.add_argument("--num_sample", default=6, help="Number of sample for dynamic prior", type=int)
    parser.add_argument('--lambda_t', type=float, default=2)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_improved_ddpm")
    parser.add_argument("--simplex_value", type=float, default=5.0)
    parser.add_argument("--clip_sample", type=bool, default=False, help="Whether to clip predicted sample between -1 and 1 for numerical stability in the noise scheduler.")

    args = parser.parse_args()
    set_seed(args)
    device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
    print(f'Device using: {device}')
    args.device = device

    if args.dataset.lower() == '20news':
        args.num_classes = 20
    elif args.dataset.lower() == 'numclaim':
        args.num_classes = 2
    elif args.dataset.lower() == 'trec':
        args.num_classes = 6
    elif args.dataset.lower() == 'semeval':
        args.num_classes = 9

    if args.noise_type == "llm":
        args.dataset_path = f"datasets/llm/{args.prompt_type}/{args.llm_type}"
    elif args.noise_type == "synthetic" or args.noise_type == "realworld":
        args.dataset_path = "datasets/realworld"

    print(args)

    train_data, train_sampler, train_dataloader, train_embedding, valid_data, valid_sampler, \
        valid_dataloader, valid_embedding, test_data, test_sampler, test_dataloader, test_embedding = create_dataset(args)
    train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
    train_masks = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
    train_true_labels = torch.stack([train_data[idx][2] for idx in range(len(train_data))], dim=0)
    train_noisy_labels = torch.tensor([train_data[idx][3] for idx in range(len(train_data))])
    valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
    valid_masks = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
    valid_true_labels = torch.stack([valid_data[idx][2] for idx in range(len(valid_data))], dim=0)
    valid_noisy_labels = torch.tensor([valid_data[idx][3] for idx in range(len(valid_data))])
    test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
    test_masks = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)
    test_true_labels = torch.stack([test_data[idx][2] for idx in range(len(test_data))], dim=0)

    print("==========================Stage I: Pre-trained Language Classifier Finetuning==========================")
    plc_trainer = PLC_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
    z_train, z_valid, z_test, best_plc_model, dists_list = plc_trainer.train()


    print("==========================Compute Training Dynamic Prior ==========================")
    print(z_train.shape) # (models, epochs, batch, dim)
    z_train = z_train.permute(2,0,1,3)
    B, M, N, D = z_train.shape
    z_train = z_train.reshape(B, M, N*D)
    z0_train = z_train[:, :, :D]

    z_valid = z_valid.permute(2,0,1,3)
    B2, M2, N2, D2 = z_valid.shape
    z_valid = z_valid.reshape(B2, M2, N2*D2)
    z0_valid = z_valid[:, :, :D2]

    z_test = z_test.permute(2,0,1,3)
    B3, M3, N3, D3 = z_test.shape
    z_test = z_test.reshape(B3, M3, N3*D3)
    z0_test = z_test[:, :, :D3]

    # Train noisy data detection
    dists_score_list = []
    markers_list = []
    for idx in range(dists_list.shape[0]):
        dists = dists_list[idx].squeeze()
        dists_labels = train_noisy_labels
        dists_mean = torch.mean(dists, 0)
        dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_var = torch.std(dists, 0)
        dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_score = dists_mean + dists_var
        dists_score = dists_score[:len(dists_labels)]
        markers = torch.zeros(len(dists_labels))
        number_points = int(len(dists_score) * args.noise_ratio)
        noisy_points = torch.topk(dists_score, number_points, largest=True).indices
        markers[noisy_points] = 1
        dists_score_list.append(dists_score.unsqueeze(0))
        markers_list.append(markers.unsqueeze(0))
    dists_score_list = torch.stack(dists_score_list, dim=0)
    markers_list = torch.stack(markers_list, dim=0)

    train_priors = []
    train_prior_weights = []
    train_uncertain_marker = []

    valid_priors = []

    for idx in range(M):
        knn_labels = train_noisy_labels
        knn_true_labels = train_true_labels
        # knn_z0 = z0_train[:, idx, :].squeeze()
        knn_z0 = train_embedding
        knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
        priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)
        
        train_priors.append(priors)
        train_prior_weights.append(weights)
        train_uncertain_marker.append(uncertain_marker)
    
    train_uncertain_marker = torch.stack(train_uncertain_marker, dim=0)

    dists_score_list = []
    markers_list = []
    dists_list = dists_list[:, :, B:(B+B2), :]
    for idx in range(dists_list.shape[0]):
        dists = dists_list[idx].squeeze()
        dists_labels = valid_noisy_labels
        dists_mean = torch.mean(dists, 0)
        dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_var = torch.std(dists, 0)
        dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
        dists_score = dists_mean + dists_var
        dists_score = dists_score[:len(dists_labels)]
        markers = torch.zeros(len(dists_labels))
        number_points = int(len(dists_score) * args.noise_ratio)
        noisy_points = torch.topk(dists_score, number_points, largest=True).indices
        markers[noisy_points] = 1
        dists_score_list.append(dists_score.unsqueeze(0))
        markers_list.append(markers.unsqueeze(0))
    dists_score_list = torch.stack(dists_score_list, dim=0)
    markers_list = torch.stack(markers_list, dim=0)



    for idx in range(M):
        knn_labels = valid_noisy_labels
        knn_true_labels = valid_true_labels
        knn_z0 = z0_valid[:, idx, :].squeeze()
        knn_z0 = valid_embedding
        knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
        priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)

        valid_priors.append(torch.argmax(weights, dim=-1))
    
    

    # majority vote for multiple model branch valid priors
    valid_priors = torch.stack(valid_priors)
    valid_p, freq = torch.mode(valid_priors, dim=0)
    final_votes = torch.empty(valid_priors.size(-1), dtype=torch.long)
    for idx in range(valid_priors.size(-1)):
        valid_priors_model = valid_priors[:, idx]
        counts = valid_priors_model.bincount(minlength=valid_priors_model.max() + 1)
        max_freq = counts.max()
        tied = torch.where(counts == max_freq)[0]

        if len(tied) > 1:
            final_votes[idx] = random.choice(tied.tolist())
        else:
            final_votes[idx] = valid_p[idx]

    valid_priors = final_votes


    train_priors = pad_sequence([model.transpose(0,1) for model in train_priors], batch_first=True, padding_value=-1).transpose(1,2)

    train_prior_weights = torch.stack(train_prior_weights, dim=0)

    train_priors = train_priors.permute(1,0,2)
    train_prior_weights = train_prior_weights.permute(1,0,2)
    train_uncertain_marker = train_uncertain_marker.permute(1,0)

    scaler = torch.amp.GradScaler("cuda")

    # prepare datasets for generative model
    train_dataset = TensorDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, train_true_labels, train_embedding)
    
    valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_priors, z_valid, valid_embedding)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    test_dataset = TensorDataset(test_inputs, test_masks, test_true_labels, z_test, test_embedding)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    simplex_trainer = Simplex_Trainer(args, train_dataset, valid_dataloader, test_dataloader, z_train.size(-1), best_plc_model)
    
    simplex_trainer.train()
if __name__ == "__main__": 
    install(show_locals=False)
    main()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()