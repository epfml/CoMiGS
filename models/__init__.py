import torch
from .lora import LoRALinear
from .gating import MoE, TopKGate, Router, CachedRouter, SoftmaxGate
from .model_types import *
from dataclasses import replace

LORA_MODULES = set([model_types.MOE])

def _moe_factory(config, module_factory, router):
    updated_config = replace(config, lin_type=config.moe_lin_type)
    if router is None:
        gate = _gating_factory(config)()
        router_factory = get_router_factory(config.router_type, config)
        router = router_factory(gate)
    return MoE(updated_config, router, module_factory)

def _gating_factory(config):
    if config.gating_type == model_types.TOPK:
        return lambda: TopKGate(config.device, config.straight_through, config.topk_exp)
    elif config.gating_type == model_types.SOFT:
        return lambda: SoftmaxGate(config.device)
    else:
        raise ValueError("unknown gating type {config.gating_type}")

def get_custom_module_factory(desc, config, module_factory, router = None):
    if desc == model_types.STANDARD:
        return lambda: module_factory(config), router
    elif desc == model_types.MOE:
        if router is None and config.is_cached:
            gate = _gating_factory(config)()
            router_factory = get_router_factory(config.router_type, config)
            router = router_factory(gate)
            router = CachedRouter(router)
        return lambda: _moe_factory(config, module_factory, router), router
    elif desc == model_types.EXPANDED:
        return lambda: module_factory(config, expansion=16), router
    else:
        raise ValueError("unknown model type {desc}")

def get_router_factory(desc, config):
    factories = {
        **dict.fromkeys(
            [model_types.STANDARD, model_types.LINEAR],
            lambda gate: Router(config.n_embd, config.moe_router_dim, config.expert_num, gate, noise_type= config.noise_type)),
    }
    return factories[desc]

def get_linear_factory(desc, config):
    factories = {
        **dict.fromkeys([model_types.STANDARD, model_types.LINEAR], torch.nn.Linear),
        model_types.LORA: lambda in_features, out_features, bias: LoRALinear(
            in_features,
            out_features,
            bias,
            device=config.device,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout)
    }
    return factories[desc] # get_module_factory(desc, factories)

def is_lora_fine_tuning(config):
    # return (_has_lora_module(config.mlp_type) or _has_lora_module(config.attention_type)) and config.gated_layer_id <= config.n_layer
    return config.lora_rank > 0

def expert_info(network):
    layers = [
        module
        for module in network.modules()
        if isinstance(module, torch.nn.Module)
    ]
    expert_activation_layers = [
        layer.expert_info()
        for layer in layers
        if getattr(layer, "expert_info", None)
        and layer.expert_info() is not None
    ]
    return expert_activation_layers

def reset_expert_info(network):
    layers = [
        module
        for module in network.modules()
        if isinstance(module, torch.nn.Module)
    ]
    expert_activation_layers = [
        layer.reset_expert_info()
        for layer in layers
        if getattr(layer, "reset_expert_info", None)
        and layer.reset_expert_info() is not None
    ]
    return expert_activation_layers