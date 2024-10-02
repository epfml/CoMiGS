from tqdm import tqdm
import argparse
from collab_utils.clients import GeneralClient
from collab_utils.server import Server
from models.model import GPTConfig, GPT
import numpy as np
import random
import torch 
import os
import ast
from collab_utils.collaboration_strategies import to_collaboration_strategy
from collab_utils.aggregation_strategies import to_aggregation_strategy
import wandb
from models.lora import get_ft_model

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
parser.add_argument('-en', '--expert_numbers', default='[2,2,2,2]', type=parse_list, help='Comma-separated list of number of experiments')
parser.add_argument('-k', '--topk', default=2,type=int)
parser.add_argument('-as','--collaboration_strategy',default="all", type=str)
parser.add_argument('-aggregation_strategy','--aggregation_strategy',default="default", type=str)
parser.add_argument('-bs','--batch_size',default=64,type=int)
parser.add_argument('-micro_bs','--micro_batch_size',default=8,type=int)
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
args = parser.parse_args()

assert len(args.expert_lora_ranks) == args.num_clients, f"Please specify lora rank for each client {args.expert_lora_ranks}."
assert len(args.expert_numbers) == args.num_clients, f"Please specify number of expersts for each client {args.expert_numbers}."
assert (len(set(args.expert_numbers)) == 1 and args.collaboration_strategy == "all") or \
    args.collaboration_strategy != "all", f"Different number of experts is not supported for `all` strategy: {args.expert_numbers}"
assert (all(value == 1 for value in args.expert_numbers) and len(set(args.expert_lora_ranks)) > 1) and (args.collaboration_strategy == "all" or args.collaboration_strategy == "ffalora") or \
    len(set(args.expert_lora_ranks)) == 1, \
        f"Different number of lora ranks is only supported for `all` strategy and 1 expert for each cliet: {args.collaboration_strategy}, {args.expert_numbers}, {args.expert_lora_ranks}"
# assert all(args.topk <= en for en in args.expert_numbers), "Each value in topk must be less than or equal to the corresponding value in expert_numbers"

collaboration_strategy = to_collaboration_strategy(args.collaboration_strategy)
aggregation_strategy = to_aggregation_strategy(args.aggregation_strategy)

print("is alternating updates:", args.is_alternating)
print("alter_on_train:", args.alter_gate_update_on_train)

def init_client_model(override_args):

    model = GPT.from_pretrained(args.base_model, override_args)
    model = get_ft_model(model, collaboration_strategy)
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(args.seed)

if args.wandb_log:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
print('=============== initializing clients and server')
acc_steps = args.batch_size // args.micro_batch_size
clients = {}
for i in range(args.num_clients):
    override_args = dict(
        expert_num = args.expert_numbers[i],
        lora_rank = args.expert_lora_ranks[i],
        lora_dropout = args.lora_dropout,
        topk_exp = min(args.topk,args.expert_numbers[i]),
        load_balancing_lambda = args.lb_lambda,
        pruning_lambda = args.p_lambda,
        pruning_strength = args.pruning_strength,
        pruning = args.is_pruning,
        expert0_importance = args.expert0_importance,
        is_no_router = args.is_no_router,
        device = args.device)
    clients[i] = GeneralClient(
        args=args,
        client_id=i,
        model=init_client_model, 
        data_path = os.path.join(args.data_path,str(args.num_clients)), 
        output_dir = args.output_dir,
        override_args = override_args)
server_override_args = dict(
    expert_num = min(args.expert_numbers),
    lora_rank = max(args.expert_lora_ranks),
    topk_exp = args.topk,
    is_no_router = args.is_no_router,
    device = args.device)
server = Server(args, GPT, config = server_override_args)

print('=============== collaborative finetuning')
for epoch in tqdm(range(args.num_global_rounds)):
    for id in range(args.num_clients):
        clients[id].synchronize(server.server_model, collaboration_strategy, aggregation_strategy, id)
        clients[id].train(acc_steps = acc_steps, local_num_steps = args.num_local_steps)
    with torch.no_grad():
        server.aggregate_parameters([clients[i].model for i in range(args.num_clients)], collaboration_strategy, aggregation_strategy, [clients[i].num_train_samples for i in range(args.num_clients)])
if args.save_model == True:
    for id in range(args.num_clients):
        out_dir = os.path.join(args.output_dir, f'client_{id}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        clients[id].save_model(out_dir)
