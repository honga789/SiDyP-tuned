import torch
import time
import math
import os
import re
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.optim import Adam
from tqdm import tqdm
from copy import deepcopy

from simplex_utils import convert_to_simplex, scale, logits_projection, self_condition_preds,\
                            adjust_learning_rate, EMA
from simplex_diff import SimplexDDPMScheduler, SimplexDiffusion
from utils import kl_div


class Simplex_Trainer:
    def __init__(self, args, train_dataset, valid_dataloader, test_dataloader, w_dim, best_plc):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.best_plc = best_plc

        self.simplex_diffs = nn.ModuleList()
        self.EMAs = [EMA(mu=0.9999) for _ in range(self.args.num_model)]

        noise_scheduler = SimplexDDPMScheduler(
            num_train_timesteps=args.train_timesteps,
            beta_schedule=args.beta_schedule,
            simplex_value=args.simplex_value,
            clip_sample=args.clip_sample,
            device=args.device,
        )
        inference_noise_scheduler = SimplexDDPMScheduler(
            num_train_timesteps=args.infer_timesteps,
            beta_schedule=args.beta_schedule,
            simplex_value=args.simplex_value,
            clip_sample=args.clip_sample,
            device=args.device,
        )

        for i in range(self.args.num_model):
            simplex_diff = SimplexDiffusion(args, noise_scheduler, inference_noise_scheduler, \
                num_classes=self.args.num_classes, w_dim=w_dim)
            simplex_diff.to(self.args.device)
            self.EMAs[i].register(simplex_diff._denoise_fn)
            self.simplex_diffs.append(simplex_diff)
        
        self.optimizer = Adam(self.simplex_diffs.parameters(), lr=self.args.diff_lr, \
                            weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)

        self.model_path = "/storage/home/hcoda1/7/lye48/p-schava6-0/weak_supervision/SiDyP/best_models/"
        if self.args.noise_type == "llm":
            self.model_path += f"{self.args.noise_type}/{self.args.prompt_type}/{self.args.llm_type}/{self.args.dataset}/{self.args.seed}"
        elif self.args.noise_type == "realworld":
            self.model_path += f"{self.args.dataset}/{self.args.seed}"
        elif self.args.noise_type == "synthetic":
            self.model_path += f"{self.args.dataset}/{self.args.noise_ratio}/{self.args.syn_type}"
                            
    def sample_uncertain_y(self, y, weights, uncertain_idx):

        # filter out 0.0 in weights
        weights = [weight[weight != 0.0] for weight in weights]

        # sample a prior according to the probability
        # batch_size, model_branches
        sample_y = torch.zeros(len(weights))

        for i, weight in enumerate(weights):
            sample_index = torch.multinomial(weight.clone().detach(), num_samples=1, replacement=True)
            sample_y[i] = y[i][sample_index]

        return sample_y.to(dtype=torch.long).to(self.args.device), weights

    
    def update_dataset(self, step, update_weight):
        batch_size = self.args.train_batch_size
        inputs, y, weights, uncertain, labels, true_labels, x_embed = self.train_dataset[:]
        weights[step*batch_size:(step+1)*batch_size] = update_weight
        self.train_dataset = TensorDataset(inputs, y, weights, uncertain, labels, true_labels, x_embed)
        
    def update_model(self, epoch):
        self.simplex_diffs.train()
        train_sampler = SequentialSampler(self.train_dataset)
        train_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if epoch <= self.args.warmup_epochs*self.args.diff_epochs:
            self.lambda_t = 0
        else:
            self.lambda_t = self.args.lambda_t
        
        num_steps = 0
        total_loss = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train diffusion epoch {epoch}", ncols=120) as pbar:
            for step, data_batch in pbar:
                num_steps += 1
                adjust_learning_rate(self.optimizer, step/len(train_loader)+epoch, warmup_epochs=self.args.warmup_epochs*self.args.diff_epochs, n_epochs=self.args.diff_epochs, lr_input=self.args.diff_lr)

                [w, y, weights, uncertain_markers, noisy_y, true_labels, x_embed] = data_batch

                y = y.to(self.args.device)
                weights = weights.to(self.args.device)
                uncertain_markers = uncertain_markers.to(self.args.device)
                noisy_y = noisy_y.to(self.args.device)
                w = w.to(self.args.device).squeeze()
                n = w.size(0)
                x_embed = x_embed.to(self.args.device).squeeze()

                total_diff_loss = 0.0
                total_reg_loss = 0.0
                prob_list = torch.zeros(self.args.num_model, n, self.args.num_classes)

                for model_i in range(self.args.num_model):
                    y_model = y[:, model_i, :]
                    weights_model = weights[:, model_i, :]
                    uncertain_markers_model = uncertain_markers[:, model_i]
                    w_model = w[:, model_i, :]

                    uncertain_idx = torch.where(uncertain_markers_model==True)[0]
                    certain_idx = torch.where(uncertain_markers_model==False)[0]
                    
                    current_y_model = torch.zeros(y_model.size(0), device=self.args.device, dtype=torch.long) * -1
                    if uncertain_idx.numel() != 0 and (epoch >= self.args.warmup_epochs*self.args.diff_epochs):
                        uncertain_y_batch = y_model[uncertain_idx]

                        # filter ourt pad value -1
                        uncertain_y_batch = [y[y!=-1] for y in uncertain_y_batch]

                        uncertain_weights_batch = weights_model[uncertain_idx]

                        for sample_i in range(self.args.num_sample):
                            # sample the prior based on uncertain prior weight
                            sample_y, uncertain_weights = self.sample_uncertain_y(uncertain_y_batch, uncertain_weights_batch, uncertain_idx)

                            with torch.amp.autocast('cuda'):
                                self.simplex_diffs[model_i].eval()
                                outputs = self.simplex_diffs[model_i].reverse_t(noisy_y[uncertain_idx], \
                                        w[uncertain_idx, model_i, :], x_embed[uncertain_idx, :], \
                                        generator=torch.Generator(device=self.args.device).manual_seed(self.args.seed))
    
                            # Update Uncertain Weights based on Model Feedback
                            # =================================================
                            pred_labels = torch.argmax(outputs.simplex, dim=-1)
                            correct_uncertain = 0
                            for (i, idx) in enumerate(uncertain_idx):
                                pred_y = pred_labels[i]
                                true_uncertain_y = true_labels[idx]
                                if pred_y == true_uncertain_y:
                                    correct_uncertain += 1
                                uncertain_y = uncertain_y_batch[i].clone().detach()
                                # (num_class,)
                                uncertain_weights = weights_model[idx]

                                if pred_y in uncertain_y:
                                    # update the uncertain y weights
                                    pred_weights = uncertain_weights[pred_y].clone()
                                    update_pred_weights = pred_weights + ((1-pred_weights) / self.args.num_sample)
                                    uncertain_weights[pred_y] = update_pred_weights
                                    # normalize updated weights
                                    uncertain_weights = uncertain_weights / (update_pred_weights + (1-pred_weights))

                                    # update in original weights
                                    weights[idx, model_i] = uncertain_weights.to(self.args.device)
                            # =================================================

                        self.update_dataset(step, weights)
                        # Evaluate after weight updating
                        sample_y, uncertain_weights = self.sample_uncertain_y(uncertain_y_batch, weights[uncertain_idx,model_i, :], uncertain_idx)
                        
                        # update y_model for sample uncertain y
                        current_y_model[uncertain_idx] = sample_y.to(torch.long)

                    # certain data points
                    if certain_idx.numel() != 0:
                        certain_y_batch = y_model[certain_idx]
                        # filter out pad value -1
                        certain_y_batch = torch.tensor([y[y != -1] for y in certain_y_batch], device=self.args.device)
                        current_y_model[certain_idx] = certain_y_batch
                        current_y_model = current_y_model[current_y_model != -1]

                    with torch.amp.autocast('cuda'):
                        self.simplex_diffs.train()
                        weights_model = torch.tensor([weights_model[i, current_y_model[i]] for i in range(weights_model.size(0))], device=self.args.device)
                        current_y_model = nn.functional.one_hot(current_y_model, num_classes=self.args.num_classes)

                        if epoch >= self.args.warmup_epochs*self.args.diff_epochs:                                    
                            loss, logits = self.simplex_diffs[model_i].forward_t(current_y_model, w_model, x_embed, noisy_y)
                            prob = torch.softmax(logits, dim=-1)
                            weighted_ce_loss = torch.matmul(weights_model, loss.double())
                            diff_loss = torch.mean(weighted_ce_loss)
                
                            prob_list[model_i] = prob
                        else:
                            loss, logits = self.simplex_diffs[model_i].forward_t(current_y_model[certain_idx], w_model[certain_idx, :], x_embed[certain_idx, :], noisy_y[certain_idx])
                            prob = torch.softmax(logits, dim=-1)
                            weighted_ce_loss = torch.matmul(weights_model[certain_idx], loss.double())
                            diff_loss = torch.mean(weighted_ce_loss)
                            diff_loss = torch.mean(loss)
                    
                    total_diff_loss += diff_loss
                
                if epoch >= self.args.warmup_epochs*self.args.diff_epochs:
                    avg_pred = torch.mean(prob_list, dim=0)

                    for model_i in range(self.args.num_model):
                        temp_pred = prob_list[model_i]
                        reg_loss = kl_div(avg_pred.squeeze(), temp_pred.squeeze())
                        total_reg_loss += torch.sum(reg_loss)

                    diff_loss, reg_loss = total_diff_loss/self.args.num_model, total_reg_loss/self.args.num_model
                    loss = diff_loss + self.lambda_t * reg_loss
                else:
                    loss = total_diff_loss / self.args.num_model

                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.simplex_diffs[model_i].parameters(), 1.0)
                self.optimizer.step()

                for model_i in range(self.args.num_model):
                    self.EMAs[model_i].update(self.simplex_diffs[model_i]._denoise_fn)
                
                total_loss += loss
            
            return total_loss / num_steps
            
    def evaluate(self, epoch, eval_loader):
        self.best_plc.eval()
        self.simplex_diffs.eval()
        start = time.time()
        with torch.no_grad():
            correct = 0
            plm_correct = 0
            all_sample = 0
            
            for test_batch_idx, data_batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f'Simplex Diffusion Sampling...', ncols=100):
                input_ids, input_mask, target, w, x_embed = data_batch 
                outputs = self.best_plc(input_ids, attention_mask=input_mask)
                w = w.squeeze().to(self.args.device)
                target = target.squeeze().to(self.args.device)

                
                logits = [output[0] for output in outputs]
                p_y_tilde = [F.softmax(logit, dim=-1).detach().cpu() for logit in logits]
                avg_p_y_tilde = torch.mean(torch.stack(p_y_tilde, dim=0), 0)
                _, plm_pred_labels = torch.max(avg_p_y_tilde, dim=-1)

                p_y_y_tilde_list = []
                for model_i in range(self.args.num_model):
                    p_y_bar_x_y_tilde = torch.zeros(target.size(0), self.args.num_classes, self.args.num_classes).to(self.args.device)
                    for label in range(self.args.num_classes):
                        labels = torch.ones(target.size(0)) * label
                        outpus = self.simplex_diffs[model_i].reverse_t(labels.to(self.args.device), w[:,model_i,:], x_embed, generator=torch.Generator(device=self.args.device).manual_seed(self.args.seed))
                        prob = torch.softmax(outpus.simplex, dim=1)
                        p_y_bar_x_y_tilde[:,:,label] = prob

                    
                    # P(y|y^,x)*P(y^|x)=P(y,y^|x)
                    p_y_expansion = p_y_tilde[model_i].squeeze().reshape(w.size(0), 1, self.args.num_classes).repeat([1, self.args.num_classes, 1])
                    p_y_y_tilde = p_y_bar_x_y_tilde.cpu().detach() * p_y_expansion  # batch*class*label
                    p_y_y_tilde_list.append(p_y_y_tilde)
                if self.args.num_model == 1:
                    p_y_y_tilde_final = p_y_y_tilde_list[-1].squeeze()
                else:
                    p_y_y_tilde_final = torch.stack(p_y_y_tilde_list, dim=0).mean(0)
                _, pred_labels = torch.max(torch.sum(p_y_y_tilde_final, dim=2), dim=1)

                correct += torch.sum(pred_labels==target.detach().cpu()).item()
                plm_correct += torch.sum(plm_pred_labels==target.detach().cpu()).item()
                all_sample += w.size(0)

        print(f'time cost for sampling: {time.time() - start}')

        acc = 100 * correct / all_sample
        plm_acc = 100 * plm_correct / all_sample
        return acc, plm_acc


    def train(self):
        best_valid_acc = 0.0
        print('Simplex Diffusion Training Start')
        for epoch in range(self.args.diff_epochs):
            train_loss = self.update_model(epoch)
            
            # validation
            if (epoch % 1 == 0 and epoch >= self.args.warmup_epochs*self.args.diff_epochs) or epoch == self.args.diff_epochs-1:
                valid_acc, plm_acc  = self.evaluate(epoch, self.valid_dataloader)
                if valid_acc > best_valid_acc:
                    print("Model Saved!")
                    best_valid_acc = max(best_valid_acc, valid_acc)
                    self.best_diff_model = deepcopy(self.simplex_diffs)
                print(f"Epoch {epoch}: PLM valid acc: {plm_acc}, Denoising valid acc: {valid_acc}")

        self.test() 
    
    def test(self):
        del self.simplex_diffs
        self.simplex_diffs = self.best_diff_model

        test_acc, plm_acc = self.evaluate(self.args.diff_epochs, self.test_dataloader)
        print(f"PLM test acc: {plm_acc}, Denoising test acc: {test_acc}")
            
            