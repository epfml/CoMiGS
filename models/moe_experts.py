from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import torch
from torch import nn
from torch.distributions.normal import Normal

from transformers.utils import (
    ModelOutput
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.lora_2 import LoRALinear2, to_lora_config

@dataclass
class ExpertInfo:
    expert_activations: torch.Tensor
    expert_average_scores: torch.Tensor


@dataclass
class CalculatorOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    pruning_loss: Optional[torch.FloatTensor] = None


@dataclass
class BaseMoEModelOutputWithPast(ModelOutput):
    """
    Args:
        num_dropped_tokens: layer idx to the number of dropped tokens
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[torch.LongTensor] = None
    gate_importance: Optional[torch.FloatTensor] = None
    expert2tokens: Optional[dict] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[torch.LongTensor]] = None
    gate_importance: Optional[Tuple[torch.FloatTensor]] = None
    expert2tokens: Optional[dict] = None


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[torch.LongTensor] = None
    gate_importance: Optional[torch.FloatTensor] = None
    expert2tokens: Optional[dict] = None

class TopKBalancedNoisyGate(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
        expert0_importance=None,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.per_token_router = self.get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight
        self.expert0_importance = expert0_importance

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def get_gate_network(self, gate_network_type, input_size, num_experts):
        gate_network_type = gate_network_type.lower()

        if gate_network_type == "linear":
            gate_network = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(gate_network.weight)
        elif gate_network_type == "mlp":
            gate_network = torch.nn.Sequential(
                torch.nn.Linear(input_size, num_experts, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts, num_experts, bias=False),
            )
        else:
            raise ValueError(f"Unexpected gate network type: {gate_network_type}.")

        return gate_network

    def reset_gate_network(self):
        self.per_token_router = self.get_gate_network(
            self.gate_network_type, self.input_size, self.num_experts
        )

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x):
        logits_gate = self.per_token_router(x)

        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control
            logits = logits_gate + logits_noise
        else:
            logits = logits_gate

        top_logits, top_indices = logits.topk(
            min(self.num_selects + 1, self.num_experts), dim=1
        )  # select the top (k+1) experts
        top_k_logits = top_logits[:, : self.num_selects]
        top_k_indices = top_indices[:, : self.num_selects]
        top_k_scores = (
            self.softmax(top_k_logits.to(torch.float32))
            if self.use_softmax
            else top_k_logits
        )
        top_k_scores = top_k_scores.to(logits.dtype)

        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(
            dim=1, index=top_k_indices, src=top_k_scores
        )  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = (
                    torch.arange(batch_size, device=x.device) * m + self.num_selects
                )
                threshold_if_in = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
                )
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
                )
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf(
                    (logits_gate - threshold_if_in) / noise_control
                )
                prob_if_out = self.normal.cdf(
                    (logits_gate - threshold_if_out) / noise_control
                )
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn(
                        'Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                        'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.'
                    )
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            expert_factors = torch.ones_like(importance)
            if self.expert0_importance is not None:
                expert_factors[1:] *= (expert_factors.shape[0] - 1) / (1 - self.expert0_importance)
                expert_factors[0] /= self.expert0_importance
                expert_factors /= expert_factors.shape[0]

            balance_loss = self.cv_squared(importance * expert_factors) + self.cv_squared(load * expert_factors)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

class UniversalCalculator(nn.Module):
    def __init__(
        self,
        experts: nn.ModuleList,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        is_tracking=False,
        pruning_lambda = 0.05,
        pruning_strength = 0.0,
        pruning = False,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        # TODO (zhutong): use vmap to boost the training efficiency
        # self.experts_vmap = torch.vmap(self.experts)
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = len(experts)
        self.mlp_norm = None

        self.expert_activation = None
        self.expert_scores_sum = 0
        self.expert_counts = 1
        self.is_tracking = is_tracking
        
        self.pruning_lambda = pruning_lambda
        self.pruning = pruning
        self.pruning_strength = pruning_strength

    def reset_experts(self):
        self.experts.reset_parameters()

    def forward(
        self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
    ) -> CalculatorOutput:
        batch_size = topK_indices.size(0)  # topK_indices: (bsz*seq_len, num_selects)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = torch.arange(
            batch_size, device=topK_scores.device
        ).repeat_interleave(num_selects)
        _, index_sorted_topK_indices = topK_indices.sort(0)

        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(
                minlength=self.num_experts
            ).tolist()

        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)

        if self.is_tracking:
            self._track_expert_activations(topK_indices, self.num_experts)
            self._track_expert_score(topK_scores, topK_indices, self.num_experts)

        expert_outputs = [
            self.experts[i](split_x[i])
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        # (bsz*seq_len*num_selects, hidden_size)
        cat_expert_outputs = torch.cat(expert_outputs, 0)
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor,
                )
                # cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * 1.0)
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)

        if not self.multiply_gate_scores:
            y = y / self.num_experts

        pruning_loss = 0
        if self.pruning:
            for expert in self.experts:
                if expert.gate_proj.is_lora():
                    index_A = int(self.pruning_strength * expert.gate_proj.lora_A.shape[0])
                    index_B = int(self.pruning_strength * expert.gate_proj.lora_B.shape[1])

                    norm_A = torch.norm(expert.gate_proj.lora_A[index_A:, :])
                    norm_B = torch.norm(expert.gate_proj.lora_B[:, index_B:])

                    pruning_loss += norm_A * norm_B

                if expert.up_proj.is_lora():
                    index_A = int(self.pruning_strength * expert.up_proj.lora_A.shape[0])
                    index_B = int(self.pruning_strength * expert.up_proj.lora_B.shape[1])

                    norm_A = torch.norm(expert.up_proj.lora_A[index_A:, :])
                    norm_B = torch.norm(expert.up_proj.lora_B[:, index_B:])

                    pruning_loss += norm_A * norm_B

                if expert.down_proj.is_lora():
                    index_A = int(self.pruning_strength * expert.down_proj.lora_A.shape[0])
                    index_B = int(self.pruning_strength * expert.down_proj.lora_B.shape[1])

                    norm_A = torch.norm(expert.down_proj.lora_A[index_A:, :])
                    norm_B = torch.norm(expert.down_proj.lora_B[:, index_B:])

                    pruning_loss += norm_A * norm_B

            # if pruning_loss.item() > 0.0:
            #     breakpoint()
            pruning_loss *= self.pruning_lambda
            # if pruning_loss.item() > 0.0:
            #     breakpoint()
        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0), pruning_loss=pruning_loss)
    
    def _track_expert_activations(self, indices, num_experts):
        num_used_experts = 1 if indices.dim() == 1 else indices.size(1)
        if self.expert_activation is None:
            self.expert_activation = torch.zeros((num_experts,), device=indices.device, requires_grad=False)
        for j in range(num_used_experts):
            index = indices if indices.dim() == 1 else indices[:, j]
            self.expert_activation[:].scatter_add_(0, index, torch.ones_like(index, device=self.expert_activation.device).float().detach())

    def _track_expert_score(self, topK_scores, topK_indices, num_experts):
        if self.expert_scores_sum is None:
            self.expert_scores_sum = torch.zeros(num_experts, device=topK_scores.device, dtype=torch.float, requires_grad=False)
            self.expert_counts = torch.zeros(num_experts, device=topK_scores.device, dtype=torch.float, requires_grad=False)

            self.expert_scores_sum.scatter_add_(0, topK_indices.detach().view(-1), topK_scores.detach().view(-1).float())
            self.expert_counts.scatter_add_(0, topK_indices.detach().view(-1), torch.ones_like(topK_scores).detach().view(-1).float())

    def reset_expert_info(self):
        self.expert_activation = None
        self.expert_scores_sum = 0
        self.expert_counts = 1

    def expert_info(self):
        return ExpertInfo(
            expert_activations=self.expert_activation,
            expert_average_scores=self.expert_scores_sum / self.expert_counts)

    
class MoELayer(nn.Module):
    def __init__(self, experts, config, **kwargs):
        super(MoELayer, self).__init__()
        self.num_experts = config.num_experts
        self.router_gate = TopKBalancedNoisyGate(
            config.hidden_size,
            config.num_experts,
            config.num_selects,
            gate_network=kwargs.get("gate_network", "mlp"),
            use_softmax=kwargs.get("gate_use_softmax", True),
            use_balance=kwargs.get("gate_use_balance", True),
            balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
            add_noise=kwargs.get("gate_add_noise", True),
            noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            expert0_importance=kwargs.get("expert0_importance", None),
        )
        self.calculator = UniversalCalculator(
            experts,
            multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            score_scale_factor=kwargs.get("score_scale_factor", 1.0),
            pruning_lambda = config.pruning_lambda,
            pruning = config.pruning,
            pruning_strength = config.pruning_strength,
        )
        self.input_size = config.hidden_size
        self.output_size = config.hidden_size

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)
        router_gate_output: dict = self.router_gate(x)
        calc_outs: CalculatorOutput = self.calculator(x, **router_gate_output)
        y = calc_outs.hidden_states
        y = y.reshape(original_shape + (self.output_size,))

        balance_loss = router_gate_output.get("balance_loss") + calc_outs.get("pruning_loss")

        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=balance_loss,
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=router_gate_output.get("load", torch.tensor(-1)),
            gate_importance=router_gate_output.get("importance", torch.tensor(-1)),
        )

    def set_num_selects(self, num_selects):
        if num_selects > self.gate.num_experts:
            raise ValueError(
                'The value of "num_selects" must satisfy "num_selects <= num_experts"!'
            )
        else:
            self.num_selects = num_selects
            self.router_gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        self.router_gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        self.router_gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        self.router_gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        self.router_gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        self.router_gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        self.calculator.score_scale_factor = score_scale_factor

    def reset_gate_network(self):
        self.router_gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()
