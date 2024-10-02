import transformers
from contextlib import nullcontext
import os
from datasets import load_dataset
import copy
import numpy as np
from collections import OrderedDict
import torch
import time
import wandb
from tqdm import tqdm
from torch.optim import lr_scheduler
from models import expert_info, reset_expert_info


def get_batch(data, seq_length, batch_size, device='cpu'):
    '''
    returns a batch of size ([seq_length, batch_size])
    '''
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    if "cuda" in torch.device(device).type:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y

class GeneralClient:
    def __init__(self, args, client_id, model, data_path, output_dir, override_args, seq_length=128):
        self.client_id = client_id
        self.local_train_data_path = os.path.join(data_path, "train_{}.bin".format(self.client_id))
        self.local_test_data_path = os.path.join(data_path, "test_{}.bin".format(self.client_id))
        self.local_valid_data_path = os.path.join(data_path, "valid_{}.bin".format(self.client_id))
        self.local_data_train = np.memmap(self.local_train_data_path, dtype=np.uint16, mode='r')
        self.num_train_samples = self.local_data_train.size
        self.local_data_test = np.memmap(self.local_test_data_path, dtype=np.uint16, mode='r')
        self.local_data_valid = np.memmap(self.local_valid_data_path, dtype=np.uint16, mode='r')
        self.output_dir = output_dir
        self.seq_length = seq_length
        self.args = args
        self.model = model(override_args).to(args.device)
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.step = 0
        self.router_step = 0
        self.expert_optimizer, self.router_optimizer = self.model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            device_type=args.device,
            is_alternating=self.args.is_alternating)
        self.wandb_log = args.wandb_log
        self.num_eval_steps = args.num_eval_steps
        self.num_log_steps = args.num_log_steps
        self.num_router_update_steps = args.num_router_update_steps
        # if self.args.scheduler == "cosine":
        self.expert_scheduler = lr_scheduler.CosineAnnealingLR(self.expert_optimizer, 
              T_max=self.args.num_local_steps*self.args.num_global_rounds, eta_min=self.args.learning_rate/10)
        # self.router_scheduler = lr_scheduler.CosineAnnealingLR(self.router_optimizer, 
        #       T_max=self.args.num_local_steps*self.args.num_global_rounds, eta_min=self.args.learning_rate/10)
        # elif self.args.scheduler == "constant":
            # self.expert_scheduler = lr_scheduler.ConstantLR(self.expert_optimizer, factor=1.0)
        if self.args.is_alternating:
            self.router_scheduler = lr_scheduler.ConstantLR(self.router_optimizer, factor=1.0)
        model_params = sum(p.numel() for p in self.model.parameters())
        model_opt_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"====> model params of client {self.client_id}: {model_params}")
        print(f"----> model optimized params: for client {self.client_id}: {model_opt_params}")
        self.pruned = {}

    def synchronize(self, server_model, collaboration_strategy, aggregation_strategy, client_rank):
        for name, client_param in self.model.named_parameters():
            if collaboration_strategy.is_global_model(name):
                server_param = dict(server_model.named_parameters())[name]
                update_param = aggregation_strategy.to_client_data(name, client_rank, client_param, server_param)
 
                update_height, update_width = update_param.shape
                with torch.no_grad():
                    client_param.data.copy_(torch.zeros_like(client_param.data))
                    client_param.data[:update_height, :update_width].copy_(update_param.data)
                    if client_param.shape != update_param.shape:
                        self.pruned[name] = (update_height, update_width)

    def train(self, local_num_steps, acc_steps=4, is_train_on_val=False):
        type_ctx = nullcontext() if self.args.device == 'cpu' else torch.amp.autocast(device_type=self.args.device, dtype=torch.bfloat16)
        self.model.train()
        for step in tqdm(range(local_num_steps)):  
            ##### Update Expert Params on Training Dataset
            loss = 0.0
            loss_report = 0.0
            for _ in range(acc_steps): # gradient accumulation
                if is_train_on_val:
                    x, y = get_batch(self.local_data_valid, self.seq_length, batch_size=self.args.micro_batch_size, device=self.args.device)
                else:
                    x, y = get_batch(self.local_data_train, self.seq_length, batch_size=self.args.micro_batch_size, device=self.args.device)
                with type_ctx:
                    _, loss_bp , loss_r = self.model(x, targets=y)
                    loss += loss_bp
                    loss_report += loss_r
            loss = loss / acc_steps
            loss.backward()
            loss_report = loss_report/acc_steps
            self.expert_optimizer.step()
            self.expert_optimizer.zero_grad()
            if len(self.pruned) > 0:
                for name, param in self.model.named_parameters():
                    if name in self.pruned:
                        enforced_shape = self.pruned[name]
                        with torch.no_grad():
                            modified_param = param.clone()
                            
                            modified_param[enforced_shape[0]:, :] = 0
                            modified_param[:, enforced_shape[1]:] = 0
                            
                            param.data = modified_param
            self.expert_scheduler.step()
            
            self.step +=1

            ##### Update Gating Params on Valid Dataset
            if self.args.is_alternating:
                if self.step % self.num_router_update_steps == 0:
                    for r in range(self.args.gating_update_iters):
                        loss_router = 0.0
                        for _ in range(acc_steps): # gradient accumulation
                            if not self.args.alter_gate_update_on_train:
                                x, y = get_batch(self.local_data_valid, self.seq_length, batch_size=self.args.micro_batch_size, device=self.args.device)
                            else:
                                x, y = get_batch(self.local_data_train, self.seq_length, batch_size=self.args.micro_batch_size, device=self.args.device)
                            with type_ctx:
                                _, loss_bp , loss_bp_to_report = self.model(x, targets=y)
                                loss_router += loss_bp
                        loss_router = loss_router / acc_steps
                        loss_router.backward()
                        self.router_optimizer.step()
                        self.router_optimizer.zero_grad()
                        self.router_scheduler.step()
                        self.router_step += 1
                # if self.wandb_log and self.router_step % self.num_log_steps == 0:
                #     router_metrics = {
                #         f"client_{self.client_id}/router_loss": loss_router.item(),
                #         "router_iter": self.router_step,
                #         "router_lr": self.router_scheduler.get_lr()[0]
                #     }
                #     wandb.log(router_metrics)
                
            if self.wandb_log == True and self.step % self.num_log_steps == 0:
                metrics = {
                    f"client_{self.client_id}/train_loss": loss_report,
                    "iter": self.step,
                    "expert_lr": self.expert_scheduler.get_lr()[0],
                    "router_lr": self.router_scheduler.get_lr()[0] if self.args.is_alternating else self.expert_scheduler.get_lr()[0]
                }
                wandb.log(metrics)

            if self.wandb_log == True and self.step % self.num_log_steps == 0 and self.step % self.num_eval_steps == 0:
                _, val_loss, val_perplexity = self.eval()
                metrics = {
                    f"client_{self.client_id}/val_loss": val_loss,
                    f"client_{self.client_id}/val_ppl": val_perplexity,
                    "iter": self.step,
                }
                activations = expert_info(self.model)
                for idx, infos in enumerate(activations):
                    norm_activations = (infos.expert_activations / infos.expert_activations.sum()).detach().cpu().numpy()
                    scores = infos.expert_average_scores.detach().cpu().numpy()

                    layer_metrics = {}
                    for i in range(len(infos.expert_activations)):
                        layer_metrics[f'expert_score_{i}'] = scores[i]
                        layer_metrics[f'expert_activations_{i}'] = norm_activations[i]
                    metrics.update({f'client_{self.client_id}/layer_{idx}': layer_metrics})
                reset_expert_info(self.model)
                wandb.log(metrics)
    
    @torch.no_grad()
    def eval( self, max_num_batches = 100 ):
        type_ctx = nullcontext() if self.args.device == 'cpu' else torch.amp.autocast(device_type=self.args.device, dtype=torch.bfloat16)
        self.model.eval()
        loss_list_val, acc_list = [], []
        for _ in tqdm(range(max_num_batches)): 
            x, y = get_batch(self.local_data_test, self.seq_length, batch_size=self.args.batch_size, device=self.args.device)
            with type_ctx:
                logits, _, loss = self.model(x, targets=y)
            loss_list_val.append(loss)
            acc_list.append((logits.argmax(-1) == y).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 2.71828 ** val_loss
        print('val_loss:',val_loss, '\n',
              'val_perplexity:',val_perplexity)
        return val_acc, val_loss, val_perplexity

    def save_model(self, out_dir):
        # ft_model = get_ft_model_state(self.model.state_dict(), strategy)
        torch.save(self.model, os.path.join(out_dir, "model.pth"))

    def get_ft_model_state(self, weights, strategy):
        modified_state_dict = {}
        for key, value in weights.items():
            if strategy.is_optimized(key):
                modified_state_dict[key] = copy.deepcopy(value.detach())
            else:
                pass
        return modified_state_dict
    