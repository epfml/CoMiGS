from abc import ABC
import torch


DEFAULT = 'default'
HETLORA = 'hetlora'
FLEXLORA = 'flexlora'

def to_aggregation_strategy(strategy):
    '''
    all -- lora fine-tuning + routing param aggregation
    ft_params -- lora fine-tuning, no routing param aggregation
    '''
    if strategy == DEFAULT:
        return DefaultAggregationStrategy()
    elif strategy == HETLORA:
        return HetLoraAggregationStrategy()
    elif strategy == FLEXLORA:
        return FlexloraAggregationStrategy()
    else:
        raise ValueError(f"Collaborative fine-tune strategy is not support: {strategy}")

class AggregationStrategy(ABC):
    def __init__(self):
        super(AggregationStrategy, self).__init__()

    def init(self, updated_models, server_model, collaboration_strategy, pruning_strength, train_set_sizes):
        pass

    def to_update(self, name, client_data, client_rank):
        pass
  
    def to_client_data(self, name, client_rank, client_data, server_param):
        pass

class DefaultAggregationStrategy(AggregationStrategy):
    def __init__(self):
        super(DefaultAggregationStrategy, self).__init__()
        self._weight = None

    def init(self, updated_models, _ignore1, _ignore2, _ignore3, _ignore4):
        self._weight = 1 / len(updated_models)

    def to_update(self, _ignore1, client_data, _ignore3):
        return self._weight * client_data
    
    def to_client_data(self, _ignore1, _ignore2, client_data, server_param):
        client_dims = client_data.shape
        return server_param[:client_dims[0], :client_dims[1]].data.clone()
    
class FlexloraAggregationStrategy(AggregationStrategy):
    def __init__(self):
        super(FlexloraAggregationStrategy, self).__init__()
        self.all_updated_weights = None

    def init(self, updated_models, _ignore1, collaboration_strategy, _ignore2, train_set_sizes):
        self._weights = train_set_sizes
        self.all_updated_weights = []
        for _, client_model in enumerate(updated_models):
            updated_weights = {}
            param_dict = { name : param.data.clone() for name, param in client_model.named_parameters()}
            for name, client_param in client_model.named_parameters():
                if collaboration_strategy.is_global_model(name) and "lora_A" in name and "mlp" in name:
                    rank = min(client_param.shape)
                    merge_rate = 16 / rank
                    nameA = name
                    nameB = name.replace("lora_A", "lora_B")
                    low_rank_w = param_dict[nameB] @ client_param * merge_rate                
                    updated_weights[nameA] = low_rank_w.data.clone()
                    updated_weights[nameB] = low_rank_w.data.clone()
            self.all_updated_weights.append(updated_weights)

    def to_update(self, name, client_data, client_rank):
        if self.all_updated_weights is not None and name in self.all_updated_weights[client_rank]:
            return self.all_updated_weights[client_rank][name] * self._weights[client_rank] / sum(self._weights)
        else:
            return 1 / len(self._weights) * client_data
    
    def to_client_data(self, _ignore1, _ignore2, client_data, server_param):
        client_dims = client_data.shape
        return server_param[:client_dims[0], :client_dims[1]].data.clone()
    
class HetLoraAggregationStrategy(AggregationStrategy):
    def __init__(self):
        super(HetLoraAggregationStrategy, self).__init__()
        self.adaptive_weights = {}
        self.ranks = None
        self._weight = None

    def _prune_zero_rows_cols(self, tensor):
        if tensor.sum() == 0:
            breakpoint()

        non_zero_rows = tensor.abs().sum(dim=1) != 0
        pruned_tensor = tensor[non_zero_rows, :]
        
        non_zero_cols = pruned_tensor.abs().sum(dim=0) != 0
        pruned_tensor = pruned_tensor[:, non_zero_cols]
        
        return pruned_tensor.data.clone()
    
    def _to_pruned_client_data(self, nameA, client_paramA_pruned, nameB, client_paramB_pruned, server_states, pruning_strength):
        pruning_dim = client_paramA_pruned.shape[0]
        num_cols_to_prune = int(pruning_strength * pruning_dim)

        client_data_last_rank_norm_A = torch.norm(client_paramA_pruned[num_cols_to_prune:, :])
        client_data_last_rank_norm_B = torch.norm(client_paramB_pruned[:, num_cols_to_prune:])

        server_state_last_rank_norm_A = torch.norm(server_states[nameA][num_cols_to_prune:client_paramA_pruned.shape[0], :])
        server_state_last_rank_norm_B = torch.norm(server_states[nameB][:, num_cols_to_prune:client_paramB_pruned.shape[1]])

        updated_clientA_data = client_paramA_pruned
        updated_clientB_data = client_paramB_pruned

        if client_data_last_rank_norm_A * client_data_last_rank_norm_B < server_state_last_rank_norm_A * server_state_last_rank_norm_B:
            updated_clientA_data = client_paramA_pruned[:num_cols_to_prune, :].data.clone()
            updated_clientB_data = client_paramB_pruned[:, :num_cols_to_prune].data.clone()
            print(f"DEBUG: Pruning Rank {client_paramA_pruned.shape} to {updated_clientA_data.shape} for {nameA}")

        return updated_clientA_data, updated_clientB_data
    
    def to_max_dim(self, param, client_rank, name):
        if not self.ranks:
            return param
        else:
            d0 = self.ranks[client_rank][name][0]
            d1 = self.ranks[client_rank][name][1]
            return param[:d0, :d1].data.clone()

    def init(self, updated_models, server_model, collaboration_strategy, pruning_strength, _):
        self._weight = 1 / len(updated_models)
        updated_ranks = []
        self.adaptive_weights = {}
        self.all_updated_weights = []
        for client_rank, client_model in enumerate(updated_models):
            ranks = {}
            updated_weights = {}
            param_dict = { name : param.data.clone() for name, param in client_model.named_parameters()}
            for name, client_param in client_model.named_parameters():
                if collaboration_strategy.is_global_model(name) and "lora_A" in name and "mlp" in name:
                    nameA = name
                    nameB = name.replace("lora_A", "lora_B")
                    client_paramA_pruned = self.to_max_dim(client_param, client_rank, nameA)
                    client_paramB_pruned = self.to_max_dim(param_dict[nameB], client_rank, nameB)
                    if client_paramA_pruned.shape[0] != client_paramB_pruned.shape[1]:
                        breakpoint()
                    updated_clientA_data, updated_clientB_data = self._to_pruned_client_data(nameA, client_paramA_pruned, nameB, client_paramB_pruned, server_model, pruning_strength)
                    updated_weights[nameA] = updated_clientA_data.data.clone()
                    updated_weights[nameB] = updated_clientB_data.data.clone()
                    ranks[nameA] = updated_clientA_data.shape
                    ranks[nameB] = updated_clientB_data.shape
            updated_ranks.append(ranks)
            self.all_updated_weights.append(updated_weights)
        self.ranks = updated_ranks

        for model_updated_weights in self.all_updated_weights:
            for name, parameter in model_updated_weights.items():
                if name not in  self.adaptive_weights:
                    self.adaptive_weights[name] = 0
                self.adaptive_weights[name] += torch.norm(parameter, p='fro')


    def to_update(self, name, client_data, client_rank):
        if name in self.adaptive_weights:
            return self.all_updated_weights[client_rank][name] * torch.norm(self.all_updated_weights[client_rank][name], p='fro') / self.adaptive_weights[name]
        else:
            return self._weight * client_data
    
    def to_client_data(self, name, client_rank, client_data, server_param):
        if not self.ranks or name not in self.ranks[client_rank]:
            client_dims = client_data.shape
            return server_param[:client_dims[0], :client_dims[1]].data.clone()
        rank = self.ranks[client_rank][name]
        return server_param[:rank[0], :rank[1]].data.clone()