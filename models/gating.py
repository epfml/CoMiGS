# Based on https://github.com/uclaml/MoE/blob/main/MoE_Visualization.ipynb

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models import model_types
from typing import List

from abc import ABC, abstractmethod

def gumbel_noise(n,k):
    unif = torch.distributions.Uniform(0,1).sample((n,k))
    g = -torch.log(-torch.log(unif))
    return g

@dataclass
class ExpertInfo:
    expert_activations: torch.Tensor
    expert_average_scores: torch.Tensor
    expert_max_scores: torch.Tensor
    expert_token_indices: List

class WSLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, gain=True, keep_init=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if gain:
            self.gain = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_std', torch.zeros(out_features, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1], keepdim=True) + 1e-5

        if self.keep_init and not self.buffer_initialized:
            with torch.no_grad():
                self.init_std.copy_(std.flatten())
            self.buffer_initialized = True
        
        if not self.keep_init:
            fan_in = weight.shape[1]
            scale_factor = 1.0 / (std * math.sqrt(fan_in))
        else:
            scale_factor = self.init_std.view(-1, 1) / std

        if self.gain is not None:
            scale_factor = scale_factor * self.gain.view(-1, 1)
        weight = scale_factor * weight  # Could also apply to outputs, note different memory impact
        return torch.nn.functional.linear(x, weight, self.bias)

class Gate(ABC):
    def __init__(self, device):
        super(Gate, self).__init__()
        self.device = device

    @abstractmethod
    def compute(self, x):
        """
        Compute the output of the gate.
        This method should be implemented by all subclasses.
        """
        pass

    def perturbation(self, shape, noise_type = model_types.NORMAL):
        """
        Noise perturbation required for gradient computation
        """
        if noise_type != model_types.NORMAL:
            raise ValueError(f"Noise type ({noise_type}) is not supported.")
        return torch.zeros(shape).to(self.device)
    
class SoftmaxGate(Gate):
    def __init__(self, device, t_func = lambda: 1.0):
        super(SoftmaxGate, self).__init__(device)
        self.device = device
        self.t_func = t_func

    def compute(self, x):
        gate_score = torch.nn.functional.softmax(x / self.t_func() , dim=-1)
        combine_tensor = gate_score[..., None, None]
        index = torch.arange(x.shape[1]).repeat(x.shape[0], 1)
        return combine_tensor, index.long(), gate_score

class TopKGate(Gate):
    def __init__(self, device, straight_through, k=1):
        super(TopKGate, self).__init__(device)
        self.k = k
        self.device = device
        self.straight_through = straight_through

    def compute(self, x):
        x = F.softmax(x, dim=-1)
        if self.k >1:
            topk_gate_scores, indices = torch.topk(
                x, self.k)
            topk_gate_scores =F.softmax(
                topk_gate_scores,
                dim=1,
                dtype=torch.float,
            ).type_as(x)
            mask = F.one_hot(indices, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (topk_gate_scores[..., None, None, None] * mask_flat[..., None, None, None] * F.one_hot(indices, x.shape[-1])[..., None, None])
            combine_tensor = combine_tensor.sum(1)
            return combine_tensor, indices, topk_gate_scores
        elif self.k == 1:
            topk_gate_scores, index = x.topk(k=self.k, dim=-1) # torch.nn.functional.softmax(x , dim=-1).topk(k=self.k, dim=-1)
            if self.straight_through:
                index_soft = F.softmax(x, dim=-1)
                index = (index - index_soft).detach() + index_soft
                index = index[:,0]
                topk_gate_scores, index = map(lambda x: x.squeeze(dim=-1), (topk_gate_scores, index))
            else:
                topk_gate_scores, index = map(lambda x: x.squeeze(dim=-1), (topk_gate_scores, index))
            
            mask = F.one_hot(index, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (topk_gate_scores[..., None, None, None] * mask_flat[..., None, None, None] * F.one_hot(index, x.shape[-1])[..., None, None])
            return combine_tensor, index, topk_gate_scores
   

    def perturbation(self, shape, noise_type):
        """
        Noise perturbation required for gradient computation
        """
        if noise_type == model_types.NORMAL:
            return torch.rand(shape).to(self.device)
        elif noise_type == model_types.GUMBEL:
            '''
            using gumbel max trick
            '''
            return gumbel_noise(shape).to(self.device)
    
class AbstractRouter(nn.Module):
    def __init__(self, gate, is_tracking, noise_type = model_types.NORMAL):
        super(AbstractRouter, self).__init__()
        self.is_tracking = is_tracking
        self.gate = gate
        self.noise_type = noise_type

        self.expert_activation = None
        self.expert_scores_sum = None
        self.expert_counts = None

    @abstractmethod
    def _forward(self, x):
        """
        Compute the output of the gate.
        This method should be implemented by all subclasses.
        """
        pass

    def init(self, idx, labels):
        """
        Initialize gating for batch
        """
        pass
    
    def forward(self, x):
        expert_scores = self._forward(x)
        combine_tensor, indices, topk_gate_scores = self.gate.compute(expert_scores + self.gate.perturbation(expert_scores.shape, self.noise_type))
        return combine_tensor, indices, topk_gate_scores

class Router(AbstractRouter):
    def __init__(self, input_dim, intermediate_dim, out_dim, gate, is_tracking=True, noise_type = model_types.NORMAL
    ):
        super().__init__(gate, is_tracking,noise_type)
        self.linear = WSLinear(input_dim, intermediate_dim, bias=False)
        self.activation = nn.ReLU(out_dim)
        self.out_dim = out_dim
        self.expert_activation = None
        self.is_tracking = is_tracking

    def _forward(self, x):
        out = self.linear(x)
        return out.mean(dim=1)

class CachedRouter(AbstractRouter):
    def __init__(self, router):
        super().__init__(router.gate, router.is_tracking, router.noise_type)
        self.router = router
        self.state = {}
        self.ROUTING_KEY = 'routing'

    def init(self, idx, labels):
        self.router.init(idx, labels)
        self.state.clear()
    
    def _forward(self, x):
        out = self.router.l3(x)
        return out.mean(dim=1)

    def forward(self, x): 
        if self.ROUTING_KEY in self.state:
            combine_tensor, index, topk_gate_scores = self.state[self.ROUTING_KEY]
            # return combine_tensor.detach(), index.detach(), expert_scores.detach()
            return combine_tensor, index, topk_gate_scores
        combine_tensor, index,  topk_gate_scores = self.router.forward(x)
        self.state[self.ROUTING_KEY] = (combine_tensor, index, topk_gate_scores)
        return combine_tensor, index, topk_gate_scores

class MoE(nn.Module):
    def __init__(self, config, router, expert_module_factory):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList()
        self.expert_num = config.expert_num
        self.is_per_token = config.is_per_token
        self.is_no_router = config.is_no_router
        for _ in range(config.expert_num):
            self.experts.append(expert_module_factory(config))
        if not self.is_per_token:
            self.router = router
        else:
            self.per_token_router = nn.Linear(config.n_embd, config.expert_num, bias=False)
        self.config = config
        self.expert_activation = None
        self.expert_scores_sum = None
        self.expert_counts = None
        self.expert_max_scores = None
        self.expert_token_indices = None

        self.expert0_importance = config.expert0_importance

    def _track_expert_activations(self, indices, num_experts):
        num_used_experts = 1 if indices.dim() == 1 else indices.size(1)
        if self.expert_activation is None:
            self.expert_activation = torch.zeros((num_experts,), device=indices.device)
        for j in range(num_used_experts):
            index = indices if indices.dim() == 1 else indices[:, j]
            self.expert_activation[:].scatter_add_(0, index, torch.ones_like(index).float())

    def _track_expert_score(self, gate_logits, num_experts):
        top_scores = nn.functional.softmax(gate_logits,dim=1)

        if self.expert_scores_sum is None:
            self.expert_scores_sum = torch.zeros(num_experts, device=top_scores.device, dtype=torch.float)
            self.expert_counts = 0

        expert_indices = torch.arange(num_experts, device=top_scores.device).view(1, -1).expand(top_scores.size(0), -1)
        self.expert_scores_sum.scatter_add_(0, expert_indices.reshape(-1), top_scores.detach().view(-1).float())
        self.expert_counts += (gate_logits.shape[0] * gate_logits.shape[1]) / num_experts

    def _track_max_expert_score(self, gate_logits, num_experts):
        top_scores = nn.functional.softmax(gate_logits,dim=1)
        top_indices = torch.argmax(gate_logits, dim=1)

        if self.expert_max_scores is None:
            self.expert_max_scores = torch.zeros(num_experts, device=top_scores.device, dtype=torch.float)
            self.expert_token_indices = [[] for _ in range(num_experts)]

        max_scores, _ = torch.max(top_scores, dim=0)
        self.expert_max_scores = torch.max(self.expert_max_scores, max_scores)

        for i, expert in enumerate(top_indices):
            self.expert_token_indices[expert.item()].append(i)

    def reset_expert_info(self):
        self.expert_activation = None
        self.expert_scores_sum = None
        self.expert_max_scores = None
        self.expert_counts = None
        self.expert_token_indices = None

    def expert_info(self):
        if self.expert_counts is None:
            return None
        return ExpertInfo(
            expert_activations=self.expert_activation,
            expert_average_scores=self.expert_scores_sum / self.expert_counts,
            expert_max_scores=self.expert_max_scores,
            expert_token_indices=self.expert_token_indices)

    def forward(self, x):

        if self.expert_num == 1:
            return self.experts[0](x)
        if self.is_no_router:
            expert_output = []
            for i, expert in enumerate(self.experts):
                expert_output.append(expert(x))
            return torch.mean(torch.stack(expert_output), dim=0)
        
        if self.is_per_token:
            x_squashed = x.view(-1, x.shape[-1])
            gate_logits = self.per_token_router(x_squashed)
            if self.config.topk_exp == 1:
                gate_logits = nn.functional.softmax(gate_logits,dim=1)
            weights, selected_experts = torch.topk(
                gate_logits, self.config.topk_exp
            )
            self._track_expert_activations(selected_experts, self.expert_num)
            self._track_expert_score(gate_logits, len(self.experts))
            # self._track_max_expert_score(gate_logits, len(self.experts))
            if self.config.topk_exp > 1:
                weights = nn.functional.softmax(
                    weights,
                    dim=1,
                    dtype=torch.float,
                ).type_as(x)
            results = torch.zeros_like(x_squashed)
            for i, expert in enumerate(self.experts):
                batch_idx, nth_expert = torch.where(selected_experts == i)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                    x_squashed[batch_idx]
                )
            return results.view_as(x)

        else:
            combine_tensor, _, _ = self.router(x)
            
            dispatch_tensor = combine_tensor.bool().to(combine_tensor)
            expert_inputs = torch.einsum('bsn,beij->ebsn', x, dispatch_tensor)

            output = []
            for i in range(self.expert_num):
                output.append(self.experts[i](expert_inputs[i]))

            output = torch.stack(output)            
            if self.config.global_routing == True:
                output = torch.einsum('beij,ebsn->bsn', dispatch_tensor, output)
            else:
                output = torch.einsum('beij,ebsn->bsn', combine_tensor, output)
            return output

    def get_load_balancing_loss(self, x):
        select, index = None, None
        if self.is_per_token:
            x_squashed = x.view(-1, x.shape[-1])
            gate_logits = self.per_token_router(x_squashed)
            if self.config.topk_exp == 1:
                gate_logits = nn.functional.softmax(gate_logits,dim=1)
            weights, index = torch.topk(
                gate_logits, self.config.topk_exp
            )
            weights_ = torch.zeros_like(gate_logits)
            weights_.scatter_(1, index, weights)
            # print('weights:', weights)
            if self.config.topk_exp >1:
                select = nn.functional.softmax(
                    weights_,
                    dim=1,
                    dtype=torch.float,
                ).type_as(x)
            else:
                select = weights_
        else:
            _, index, _ = self.router(x)
            select = F.softmax(self.router._forward(x), dim=-1)
        if self.config.topk_exp == 1:
            mask_ = F.one_hot(index.long(), self.expert_num).float()
        else:
            mask_ = F.one_hot(index.long(), self.expert_num).float().mean(1)
        mask_ = mask_.to(self.config.device)
        density = mask_.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        expert_factors = torch.ones_like(density_proxy)
        
        if self.expert0_importance != 0:
            expert_factors[1:] *= (expert_factors.shape[0] - 1) / (1 - self.expert0_importance)
            expert_factors[0] /= self.expert0_importance
            expert_factors /= expert_factors.shape[0]

        load_balancing_loss = (density_proxy * density * expert_factors).mean() * float(self.expert_num ** 2)
        return load_balancing_loss
    
    def get_pruning_loss(self):
        loss = 0
        for expert in self.experts:
            if expert.c_fc.is_lora():
                index_A = int(self.config.pruning_strength * expert.c_fc.lora_A.shape[0])
                index_B = int(self.config.pruning_strength * expert.c_fc.lora_B.shape[1])

                norm_A = torch.norm(expert.c_fc.lora_A[index_A:, :])
                norm_B = torch.norm(expert.c_fc.lora_B[:, index_B:])

                loss += norm_A * norm_B

            if expert.c_proj.is_lora():
                index_A = int(self.config.pruning_strength * expert.c_proj.lora_A.shape[0])
                index_B = int(self.config.pruning_strength * expert.c_proj.lora_B.shape[1])

                norm_A = torch.norm(expert.c_proj.lora_A[index_A:, :])
                norm_B = torch.norm(expert.c_proj.lora_B[:, index_B:])

                loss += norm_A * norm_B
        return loss

def get_moe_model(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if "router" in name or "experts" in name:
            continue
        else:
            param.requires_grad = False
    return model