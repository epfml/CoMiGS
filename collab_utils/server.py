
import torch

class Server(object):
    def __init__(self, args, model, config):
        self.server_model = model.from_pretrained(args.base_model,config).to(args.device)
        self.args = args
        self.pruning_strength = args.pruning_strength

    def aggregate_parameters(self, uploaded_models, collaboration_strategy, aggregation_strategy, train_set_sizes):
        assert (len(uploaded_models) > 0)
        server_state = {}

        aggregation_strategy.init(uploaded_models, self.server_model.state_dict(), collaboration_strategy, self.pruning_strength, train_set_sizes)
        for client_rank, client_model in enumerate(uploaded_models):
            self._update_state(client_model, collaboration_strategy, aggregation_strategy, server_state, client_rank)
        self.load_adjusted_state_dict(server_state)

    def _svd(self, server_param, client_dims):
        U, S, V = torch.svd(server_param)
        rank = min(client_dims)
        if rank == client_dims[0]:
            return V.T[:rank, :]
        else:
            return U[:, :rank] @ torch.diag(S[:rank]) / (16 / rank)

    def load_adjusted_state_dict(self, server_state):
        new_state_dict = {}
        for name, param in self.server_model.named_parameters():
            if name in server_state:
                server_param = server_state[name]
                if server_param.shape != param.shape:
                    model_dims = list(param.shape)
                    adjusted_param = self._svd(server_param.data.clone(), model_dims)
                    new_state_dict[name] = adjusted_param
                else:
                    new_state_dict[name] = server_param.clone()
        self.server_model.load_state_dict(new_state_dict, strict=False)

    def _update_state(self, client_model, collaboration_strategy, aggregation_strategy, server_state, client_rank):
        if len(server_state) == 0:
            for name, params in client_model.named_parameters():
                if collaboration_strategy.is_global_model(name):
                    server_state[name] = torch.zeros_like(self.server_model.state_dict()[name])
                    self.update_parameters(aggregation_strategy, name, params, server_state, client_rank)
        else:
            for name, params in client_model.named_parameters():
                if collaboration_strategy.is_global_model(name):
                    self.update_parameters(aggregation_strategy, name, params, server_state, client_rank)

    def update_parameters(self, aggregation_strategy, name, params, server_state, client_rank):
        update = aggregation_strategy.to_update(name, params, client_rank)
        client_dims = update.shape
        if update.shape[0] > self.server_model.state_dict()[name].shape[0] or update.shape[1] > self.server_model.state_dict()[name].shape[1]:
            server_state[name] = torch.zeros_like(update)
        server_state[name][:client_dims[0], :client_dims[1]] += update
