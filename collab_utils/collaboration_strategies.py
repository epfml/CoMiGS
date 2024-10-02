from abc import ABC, abstractmethod

LORA = 'lora'
GATE = 'per_token_router'
EXPERT_0 = 'experts.0.'
EXPERT_1 = 'experts.1.'

def to_collaboration_strategy(strategy):
    '''
    all -- lora fine-tuning + routing param aggregation
    ft_params -- lora fine-tuning, no routing param aggregation
    '''
    if strategy == 'all':
        return AllCollaborationStrategy()
    elif strategy == 'expert_0':
        return Expert0CollaborationStrategy()
    elif strategy == 'ffalora':
        return Expert0CollaborationStrategy()
    elif strategy == 'experts':
        return ExpertCollaborationStrategy()
    else:
        raise ValueError(f"Collaborative fine-tune strategy is not support: {strategy}")

class CollaborationStrategy(ABC):
    def __init__(self):
        super(CollaborationStrategy, self).__init__()

    def is_optimized(self, name):
        """
        Returns whether weight is optimized
        """
        return any(infix in name for infix in self._optimized())

    def is_client_specific(self, name):
        """
        Returns whether weight is client specific
        """
        return any(infix in name for infix in self._client_specific())
    
    @abstractmethod
    def _client_specific(self):
        pass

    @abstractmethod
    def _optimized(self):
        pass

    def is_global_model(self, name):
        """
        Returns whether weight is part of the global model
        """
        return self.is_optimized(name) and not self.is_client_specific(name)
    
class AllCollaborationStrategy(CollaborationStrategy):
    def __init__(self):
        super(AllCollaborationStrategy, self).__init__()
        self.OPT_PARAMS = [ LORA,  GATE ]
        self.name = "all"
    
    def _client_specific(self):
        return []

    def _optimized(self):
        return self.OPT_PARAMS

class ExpertCollaborationStrategy(CollaborationStrategy):
    def __init__(self):
        super(ExpertCollaborationStrategy, self).__init__()
        self.OPT_PARAMS = [ LORA,  GATE ]
        self.CLIENT_SPECIFIC = [ GATE ]
        self.name = "experts"
    
    def _client_specific(self):
        return self.CLIENT_SPECIFIC

    def _optimized(self):
        return self.OPT_PARAMS
    
    def is_client_specific(self, name):
        return any(infix in name for infix in self._client_specific())
    
class FAALoRACollaborationStrategy(CollaborationStrategy):
    def __init__(self):
        super(AllCollaborationStrategy, self).__init__()
        self.OPT_PARAMS = [ "lora_B",  GATE ]
        self.name = "all"
    
    def _client_specific(self):
        return []

    def _optimized(self):
        return self.OPT_PARAMS
    
class Expert0CollaborationStrategy(CollaborationStrategy):
    def __init__(self):
        super(Expert0CollaborationStrategy, self).__init__()
        self.CLIENT_SPECIFIC = [ GATE ]
        self.OPT_PARAMS = [ LORA ] + self.CLIENT_SPECIFIC
        self.name = "expert_0"

    def _client_specific(self):
        return self.CLIENT_SPECIFIC

    def _optimized(self):
        return self.OPT_PARAMS

    def is_client_specific(self, name):
        """
        Returns whether weight is client specific, 
        should return Export_X (X>=1) LoRA weights and Gating weights
        """
        return any(infix in name for infix in self.OPT_PARAMS) and not EXPERT_0 in name and not "attn" in name