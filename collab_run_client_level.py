from tqdm import tqdm
import argparse
from collab_utils.clients import GeneralClient
from collab_utils.server import Server
from models.model import GPT
import torch.nn.functional as F
from typing import Tuple
from argparse import Namespace
import numpy as np
import random
import torch 
from torch import nn, Tensor
import os
import ast
from collab_utils.collaboration_strategies import to_collaboration_strategy
from collab_utils.aggregation_strategies import to_aggregation_strategy
import wandb
from models.lora import get_ft_model
from contextlib import nullcontext

# Define the get_device function
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, seq_length, batch_size, is_shifted, device='cpu'):
    '''
    returns a batch of size ([seq_length, batch_size])
    '''
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    if is_shifted:
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    else:
        y = x
    if "cuda" in torch.device(device).type:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def parse_list(value):
    return ast.literal_eval(value)

parser = argparse.ArgumentParser()
parser.add_argument("-gr", "--num_global_rounds", default=20, type=int)
parser.add_argument("-num_steps", "--num_local_steps", default=25, type=int)
parser.add_argument("-model_path", "--model_path", type=str)
parser.add_argument('-lr', "--learning_rate", default=2e-3, type=float)
parser.add_argument('-wd', "--weight_decay", default=1e-2, type=float)
parser.add_argument('-ds', '--dataset', default='agnews', type=str)
parser.add_argument('-data_path', '--data_path', type=str)
parser.add_argument('-nc', '--num_clients', default=4, type=int)
parser.add_argument('-device', '--device', default=get_device(), type=str)
parser.add_argument('-el', '--expert_lora_ranks', default='[8,8,8,8]', type=parse_list, help='Comma-separated list of LoRA ranks')
parser.add_argument('-en', '--expert_numbers', default='[2,2,2,2]', type=parse_list, help='Comma-separated list of number of experiments')
parser.add_argument('-k', '--topk', default=2, type=int)
parser.add_argument('-as', '--collaboration_strategy', default="all", type=str)
parser.add_argument('-aggregation_strategy', '--aggregation_strategy', default="default", type=str)
parser.add_argument('-bs', '--batch_size', default=64, type=int)
parser.add_argument('-micro_bs', '--micro_batch_size', default=64, type=int)
parser.add_argument('-wandb', '--wandb_log', action='store_true')
parser.add_argument('-wandb_proj', '--wandb_project', default="CoMoLE", type=str)
parser.add_argument('-wandb_run_name', '--wandb_run_name', default="test", type=str)
parser.add_argument('-out_dir', '--output_dir', default="../out", type=str)
parser.add_argument('-log_every', '--num_log_steps', default=1, type=int)
parser.add_argument('-eval_every', '--num_eval_steps', default=1, type=int)
parser.add_argument('-update_router_every', '--num_router_update_steps', default=1, type=int)
parser.add_argument('-seed', '--seed', default=1, type=int)
parser.add_argument('-scheduler', '--scheduler', default="cosine", type=str)
parser.add_argument('-lb_lam', '--lb_lambda', default=0.01, type=float)
parser.add_argument('-p_lam', '--p_lambda', default=0.01, type=float)
parser.add_argument('-p_strength', '--pruning_strength', default=0.99, type=float)
parser.add_argument('-is_pruning', '--is_pruning', action='store_true', help='Enable pruning if set')
parser.add_argument('-exp0_importance', '--expert0_importance', default=0.5, type=float)
parser.add_argument('-gating_update_iters', '--gating_update_iters', default=1, type=int)
parser.add_argument('-save_model', '--save_model', action='store_true')
parser.add_argument('-lora_do', '--lora_dropout', default=0.0, type=float)
parser.add_argument('-alter_on_train', '--alter_gate_update_on_train', action='store_true')
parser.add_argument('-bm', '--base_model', default="gpt2", type=str)
parser.add_argument('-is_alter', '--is_alternating', action='store_true')
parser.add_argument('-is_no_router', '--is_no_router', action='store_true')
parser.add_argument('-learning_rate_scale','--learning_rate_scale', default=1.0, type=float)
args = parser.parse_args()

# Detect number of GPUs
def get_num_gpus():
    return torch.cuda.device_count()

num_gpus = get_num_gpus()
if num_gpus == 0:
    device_type = 'cpu'
    print("CUDA not available. Using CPU.")
elif num_gpus == 1:
    device_type = 'cuda:0'
    print("Using a single GPU: cuda:0.")
elif num_gpus >= args.num_clients:
    device_type = None  # Will assign per client
    print(f"Using {num_gpus} GPUs for assigning each client to one GPU.")
else:
    device_type = None  # Will assign per client with possible multiple clients per GPU
    print(f"Using {num_gpus} GPUs to assign {args.num_clients} clients.")

assert len(args.expert_lora_ranks) == args.num_clients, f"Please specify lora rank for each client {args.expert_lora_ranks}."
assert len(args.expert_numbers) == args.num_clients, f"Please specify number of experts for each client {args.expert_numbers}."
assert (len(set(args.expert_numbers)) == 1 and args.collaboration_strategy == "all") or \
    args.collaboration_strategy != "all", f"Different number of experts is not supported for `all` strategy: {args.expert_numbers}"
assert (all(value == 1 for value in args.expert_numbers) and len(set(args.expert_lora_ranks)) > 1) and args.collaboration_strategy == "all" or \
    len(set(args.expert_lora_ranks)) == 1, \
        f"Different number of lora ranks is only supported for `all` strategy and 1 expert for each client: {args.collaboration_strategy}, {args.expert_numbers}, {args.expert_lora_ranks}"

collaboration_strategy = to_collaboration_strategy(args.collaboration_strategy)
aggregation_strategy = to_aggregation_strategy(args.aggregation_strategy)
print("is_alter:", args.is_alternating)
print('alter_on_train:', args.alter_gate_update_on_train)
print('is_no_router:', args.is_no_router)

type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

def init_client_model(override_args):
    device = override_args['device']
    if args.base_model.startswith("gpt"):
        model = GPT.from_pretrained(args.base_model, override_args)
        model = get_ft_model(model, collaboration_strategy).to(device)
    elif "llama" in args.base_model:
        from models.modeling_llama_moe_hf import LlamaMoEForCausalLM
        from models.configuration_llama_moe import LlamaMoEConfig
        model = LlamaMoEForCausalLM.from_pretrained(args.base_model, LlamaMoEConfig(**override_args))
        print(f"=====> {args.base_model}")
        model = get_ft_model(model, collaboration_strategy).to(device)     
    else:   
        raise ValueError("Unknown model type")
    return model

def init_server_model(override_args):
    if args.base_model.startswith("gpt"):
        server = Server(args, GPT, config=override_args)
    elif "llama" in args.base_model:
        from models.modeling_llama_moe_hf import LlamaMoEForCausalLM
        from models.configuration_llama_moe import LlamaMoEConfig
        server = Server(args, LlamaMoEForCausalLM, LlamaMoEConfig(**override_args))
    else:   
        raise ValueError("Unknown model type")
    return server

@torch.no_grad()
def eval_model(model: nn.Module, data_tensor: np.ndarray, sequence_length: int, batch_size: int, device: str,
             max_num_batches: int = 24) -> Tuple[float, float, float]:
    assert not model.training
    model = model.to(device)
    loss_list_val, acc_list = [], []
    for _ in range(max_num_batches):
        x, y = get_batch(data_tensor, sequence_length, batch_size, is_shifted=args.base_model.startswith("gpt"), device=device)
        with type_ctx:
            logits, _, loss_for_reporting = model(x, targets=y)
        val_loss = loss_for_reporting
        loss_list_val.append(val_loss)
        acc_list.append((logits.argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

@torch.no_grad()
def __trust_weights_ref(clients, sequence_length: int, batch_size: int, extra_args: Namespace) -> Tensor:
    trust_weights = torch.zeros((len(clients), len(clients))).cpu()
    for id1 in range(len(clients)):
        clients[id1].model.eval().to(clients[id1].device)
    for id1 in range(len(clients)):
        for id2 in range(len(clients)):
            _, _, val_perplexity = eval_model(
                clients[id2].model, 
                clients[id1].local_data_valid, 
                sequence_length, 
                batch_size,
                clients[id2].device, 
                max_num_batches=20
            )
            trust_weights[id1, id2] = val_perplexity
    return F.softmax(-trust_weights, dim=1)

def __weighted_average(clients, trust_weights: Tensor, extra_args: Namespace) -> None:
    weights = {}
    for id, client in clients.items():
        for name, param in clients[id].model.named_parameters():
            if param.requires_grad:
                if name in weights:
                    weights[name][id] = param.data.clone().cpu()
                else:
                    weights[name] = {}
                    weights[name][id] = param.data.clone().cpu()

    for idx, client in clients.items():
        for name, param in client.model.named_parameters():
            if param.requires_grad:
                val = torch.zeros_like(param).to(param.device)
                for i in range(len(clients)):
                    val += trust_weights[idx, i].to(param.device) * weights[name][i].to(param.device)
                param.data.copy_(val)

    del weights

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(args.seed)

if args.wandb_log:
    wandb.init(project=args.wandb_project, entity='ec-llm', name=args.wandb_run_name)
        
# Prepare all clients
print('=============== Initializing clients and server')

acc_steps = args.batch_size // args.micro_batch_size

# Assign GPUs to clients
if num_gpus > 0:
    if args.num_clients <= num_gpus:
        gpu_ids = list(range(args.num_clients))
    else:
        gpu_ids = [i % num_gpus for i in range(args.num_clients)]
else:
    gpu_ids = ['cpu'] * args.num_clients

clients = {}
for client_id in range(args.num_clients):
    gpu_id = gpu_ids[client_id] if num_gpus > 0 else 'cpu'
    device = f'cuda:{gpu_id}' if gpu_id != 'cpu' else 'cpu'
    
    override_args = dict(
        expert_num = args.expert_numbers[client_id],
        lora_rank = args.expert_lora_ranks[client_id],
        lora_dropout = args.lora_dropout,
        topk_exp = min(args.topk,args.expert_numbers[client_id]),
        load_balancing_lambda = args.lb_lambda,
        expert0_importance = args.expert0_importance,
        is_no_router = args.is_no_router,
        device = device)
    
    clients[client_id] = GeneralClient(
        args=args,
        client_id=client_id,
        model=init_client_model,
        data_path=os.path.join(args.data_path, str(args.num_clients)), 
        output_dir=args.output_dir,
        override_args=override_args,
        is_shifted=args.base_model.startswith("gpt"),
        dtype=np.uint16 if args.base_model.startswith("gpt") else np.uint32)


print('=============== Collaborative Fine-tuning')

for epoch in tqdm(range(args.num_global_rounds)):
    for id in range(args.num_clients):
        client = clients[id]
        client_device = client.model.device  # This is a torch.device object
        if client_device.type == 'cuda' and client_device.index is not None:
            with torch.cuda.device(client_device.index):
                client.train(acc_steps=acc_steps, local_num_steps=args.num_local_steps)
        else:
            with nullcontext():
                client.train(acc_steps=acc_steps, local_num_steps=args.num_local_steps)
   # Periodically evaluate and average weights
    with torch.no_grad():
        # Compute trust weights on CPU
       trust_weights = __trust_weights_ref(
           clients, 
           sequence_length=128, 
           batch_size=args.batch_size, 
           extra_args=args
       )
       print(trust_weights)
       __weighted_average(clients, trust_weights, args)

if args.save_model:
    for id in range(args.num_clients):
        out_dir = os.path.join(args.output_dir, f'client_{id}')
        os.makedirs(out_dir, exist_ok=True)
        clients[id].save_model(out_dir)
