from tqdm import tqdm
import argparse
from collab_utils.clients import GeneralClient
from collab_utils.server import Server
from models.model import GPTConfig, GPT
from contextlib import nullcontext
import numpy as np
import random
import torch 
import os
import ast
from collab_utils.aggregation_strategies import to_aggregation_strategy
from collab_utils.collaboration_strategies import to_collaboration_strategy
import wandb
from models.lora import get_ft_model
import copy
from functools import partial
import nevergrad as ng

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
parser.add_argument("-gr", "--num_global_rounds", default = 20, type=int)
parser.add_argument("-num_steps", "--num_local_steps", default = 25, type=int)
parser.add_argument("-model_path", "--model_path", type = str)
parser.add_argument('-lr',"--learning_rate",default=2e-3,type=float)
parser.add_argument('-wd',"--weight_decay",default=1e-2,type=float)
parser.add_argument('-ds','--dataset',default='agnews',type=str)
parser.add_argument('-data_path','--data_path',type=str)
parser.add_argument('-nc','--num_clients',default=4,type=int)
parser.add_argument('-device','--device',default="cuda",type=str)
parser.add_argument('-el', '--expert_lora_ranks', default='[8,8,8,8]', type=parse_list, help='Comma-separated list of LoRA ranks')
parser.add_argument('-en', '--expert_numbers', default='[1,1,1,1]', type=parse_list, help='Comma-separated list of number of experiments')
parser.add_argument('-k', '--topk', default=1,type=int)
parser.add_argument('-as','--collaboration_strategy',default="all", type=str)
parser.add_argument('-aggregation_strategy','--aggregation_strategy',default="default", type=str)
parser.add_argument('-bs','--batch_size',default=64,type=int)
parser.add_argument('-micro_bs','--micro_batch_size',default=64,type=int)
parser.add_argument('-wandb','--wandb_log',action='store_true')
parser.add_argument('-wandb_proj','--wandb_project',default="CoMoLE", type=str)
parser.add_argument('-wandb_run_name','--wandb_run_name',default="test", type=str)
parser.add_argument('-out_dir','--output_dir',default="../out", type=str)
parser.add_argument('-log_every','--num_log_steps',default=1, type=int)
parser.add_argument('-eval_every','--num_eval_steps',default=1, type=int)
parser.add_argument('-update_router_every','--num_router_update_steps',default=1, type=int)
parser.add_argument('-seed','--seed',default=1, type=int)
parser.add_argument('-scheduler','--scheduler', default="cosine", type=str)
parser.add_argument('-lb_lam','--lb_lambda', default=0.01, type=float)
parser.add_argument('-p_lam','--p_lambda', default=0.01, type=float)
parser.add_argument('-p_strength','--pruning_strength', default=0.99, type=float)
parser.add_argument('-is_pruning', '--is_pruning', action='store_true', help='Enable pruning if set')
parser.add_argument('-exp0_importance','--expert0_importance', default=0.9, type=float)
parser.add_argument('-gating_update_iters','--gating_update_iters', default=1, type=int)
parser.add_argument('-save_model','--save_model', action='store_true')
parser.add_argument('-lora_do','--lora_dropout', default=0.0, type=float)
parser.add_argument('-alter_on_train','--alter_gate_update_on_train', action='store_true')
parser.add_argument('-bm','--base_model', default="gpt2", type=str)
parser.add_argument('-is_alter','--is_alternating', action='store_true')
parser.add_argument('-is_no_router','--is_no_router', action='store_true')
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
assert len(args.expert_numbers) == args.num_clients, f"Please specify number of expersts for each client {args.expert_numbers}."
assert (len(set(args.expert_numbers)) == 1 and args.collaboration_strategy == "all") or \
    args.collaboration_strategy != "all", f"Different number of experts is not supported for `all` strategy: {args.expert_numbers}"
assert (all(value == 1 for value in args.expert_numbers) and len(set(args.expert_lora_ranks)) > 1) and args.collaboration_strategy == "all" or \
    len(set(args.expert_lora_ranks)) == 1, \
        f"Different number of lora ranks is only supported for `all` strategy and 1 expert for each cliet: {args.collaboration_strategy}, {args.expert_numbers}, {args.expert_lora_ranks}"

collaboration_strategy = to_collaboration_strategy(args.collaboration_strategy)
aggregation_strategy = to_aggregation_strategy(args.aggregation_strategy)
print("is_alter:", args.is_alternating)
print("alter_on_train:", args.alter_gate_update_on_train)
print("is_no_router:", args.is_no_router)

type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

def init_client_model(override_args):
    device = override_args['device']
    if args.base_model.startswith("gpt"):
        model = GPT.from_pretrained(args.base_model, override_args)
        model = get_ft_model(model, collaboration_strategy).to(device)
    elif "llama" in args.base_model:
        print(f"=====> {args.base_model}")
        from models.modeling_llama_moe_hf import LlamaMoEForCausalLM
        from models.configuration_llama_moe import LlamaMoEConfig
        model = LlamaMoEForCausalLM.from_pretrained(args.base_model, LlamaMoEConfig(**override_args))
        model = get_ft_model(model, collaboration_strategy).to(device)  
    else:   
        raise ValueError("Unknown model type")
    return model

def init_server_model(override_args):
    if args.base_model.startswith("gpt"):
        server = Server(args, GPT, config = override_args)
    elif "llama" in args.base_model:
        from models.modeling_llama_moe_hf import LlamaMoEForCausalLM
        from models.configuration_llama_moe import LlamaMoEConfig
        server = Server(args, LlamaMoEForCausalLM, LlamaMoEConfig(**override_args))
    else:   
        raise ValueError("Unknown model type")
    return server


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_local_loras_state_dict(model):
    loras = {}
    for name, module in model.named_parameters():
        if "mlp" in name and ".experts" in name and "lora" in name:
            loras[name] =  module  
    return loras

def set_model_weights(model, state_dict):
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])

set_seed(args.seed)

if args.wandb_log:
    import wandb
    wandb.init(project=args.wandb_project, entity='ec-llm', name=args.wandb_run_name)
    
# prepare all clients
print('=============== initializing clients and server')

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
        expert_num=args.expert_numbers[client_id],
        lora_rank=args.expert_lora_ranks[client_id],
        lora_dropout=args.lora_dropout,
        topk_exp=min(args.topk, args.expert_numbers[client_id]),
        load_balancing_lambda=args.lb_lambda,
        expert0_importance=args.expert0_importance,
        is_no_router=args.is_no_router,
        device=device,
    )
    
    clients[client_id] = GeneralClient(
        args=args,
        client_id=client_id,
        model=init_client_model,
        data_path=os.path.join(args.data_path, str(args.num_clients)), 
        output_dir=args.output_dir,
        override_args=override_args,
        is_shifted=args.base_model.startswith("gpt"),
        dtype=np.uint16 if args.base_model.startswith("gpt") else np.uint32)
    

server_override_args = dict(
    expert_num = min(args.expert_numbers),
    lora_rank = max(args.expert_lora_ranks),
    topk_exp = args.topk,
    is_no_router = args.is_no_router,
    device = 'cpu',
    )
server = init_server_model(server_override_args)

cached_loras = {}


print('=============== collaborative finetuning')
# stage 1 and 2
for epoch in tqdm(range(args.num_global_rounds)):
    for id in range(args.num_clients):
        client = clients[id]
        client.synchronize(server.server_model, collaboration_strategy, aggregation_strategy, id)
        client_device = client.model.device  # This is a torch.device object
        if client_device.type == 'cuda' and client_device.index is not None:
            with torch.cuda.device(client_device.index):
                client.train(acc_steps=acc_steps, local_num_steps=args.num_local_steps)
        else:
            with nullcontext():
                client.train(acc_steps=acc_steps, local_num_steps=args.num_local_steps)
        cached_loras[id] = copy.deepcopy(get_local_loras_state_dict(clients[id].model))

    with torch.no_grad():
        server.aggregate_parameters([clients[i].model for i in range(args.num_clients)], collaboration_strategy, aggregation_strategy, [clients[i].num_train_samples for i in range(args.num_clients)])

    
server_loras = get_local_loras_state_dict(server.server_model)


# stage 3
# for each client, find the optimal [w1, w2]
# the function is adopted from LoRAHub code base, which FDLoRA is based on.

def get_score(weights, model, cache, example_dataset, batch_size, device, get_regular):
    # the composed lora state dict
    final_state_dict = {}
    keys = cache.keys()
    client_loras = cache
    for key in keys:
        final_state_dict[key] = weights[0] * client_loras[key] + weights[1] * server_loras[key]

    # reload the model with the new adapter config
    set_model_weights(model,final_state_dict)
        
    # minimize the metric
    x, y = get_batch(example_dataset, 128, batch_size, is_shifted=args.base_model.startswith("gpt"), device=device)
    type_ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    loss = 0.0
    with type_ctx:
        model.to(device)
        _, loss_bp , loss_bp_to_report = model(x, targets=y)
        loss += loss_bp
    model.to('cpu')
    # L1 regularization term
    loss = loss.item()
    metric_val = loss + get_regular(weights)
    return metric_val



for id in range(args.num_clients):
    get_score_partial = partial(get_score, 
                                model=clients[id].model, 
                                cache=cached_loras[id],
                                example_dataset=clients[id].local_data_valid,
                                batch_size=64,
                                get_regular=default_l1_regularization,
                                device=clients[id].device)
    instrum = ng.p.Array(
        init=[0.0] * 2,
        upper=[1.0] * 2,
        lower=[-1.0] * 2,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=10)
    recommendation = optimizer.minimize(get_score_partial, verbosity=0)
    final_state_dict = {}
    keys = server_loras.keys()
    client_loras = cached_loras[id]
    weights = recommendation.value
    for key in keys:
        final_state_dict[key] = weights[0] * client_loras[key] + weights[1] * server_loras[key]
    set_model_weights(clients[id].model, final_state_dict)

# evaluate 
for id in range(args.num_clients):
    clients[id].model.to(clients[id].device)
    clients[id].eval()

if args.save_model == True:
    for id in range(args.num_clients):
        out_dir = os.path.join(args.output_dir, f'client_{id}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        clients[id].save_model(out_dir)