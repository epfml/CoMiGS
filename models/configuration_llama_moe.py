from transformers.configuration_utils import PretrainedConfig


class LlamaMoEConfig(PretrainedConfig):
    model_type = "llama_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=32,
        num_hidden_layers=16,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        attention_bias=False,
        attention_dropout=0.0,
        is_no_router=False,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        use_bfloat16=True,
        # -------- moe expert configs --------
        expert_num=2,
        topk_exp=2,
        size_experts=None,
        # -------- moe gate configs --------
        gate_type="TopKBalancedNoisyGate",
        gate_network="mlp",
        gate_use_softmax=True,
        gate_use_balance=True,
        lb_lambda=1e-2,
        gate_add_noise=True,
        # TopKBalancedNoisyGate
        gate_noise_epsilon=1e-2,
        # -------- moe calculator configs --------
        calculator_type="UniversalCalculator",
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        add_weight_norm=False,
        # -------- LoRA --------
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # SwitchDropTokenCalculator
        drop_tokens=True,
        dropped_padding="zero",
        capacity_factor=1.25,
        expert0_importance=0.5,
         # -------- HetLoRA --------
        pruning=False,
        pruning_lambda=0.01,
        pruning_strength=0.99,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_bfloat16 = use_bfloat16

        self.num_experts = expert_num
        self.num_selects = topk_exp
        self.size_experts = size_experts

        self.gate_type = gate_type
        self.gate_network = gate_network
        self.gate_use_softmax = gate_use_softmax
        self.gate_use_balance = gate_use_balance
        self.gate_balance_loss_weight = lb_lambda
        self.gate_add_noise = gate_add_noise
        self.gate_noise_epsilon = gate_noise_epsilon
        self.expert0_importance = expert0_importance
        self.is_no_router = is_no_router

        self.calculator_type = calculator_type
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.add_weight_norm = add_weight_norm

        self.lora_rank=lora_rank
        self.lora_alpha=lora_alpha
        self.lora_dropout=lora_dropout

        self.drop_tokens = drop_tokens
        self.dropped_padding = dropped_padding
        self.capacity_factor = capacity_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.pruning = pruning
        self.pruning_lambda = pruning_lambda
        self.pruning_strength = pruning_strength

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        # if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
        #     raise ValueError(
        #         "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
        #         f"got {self.rope_scaling}"
        #     )
        # rope_scaling_type = self.rope_scaling.get("type", None)
        # rope_scaling_factor = self.rope_scaling.get("factor", None)
        # if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
        #     raise ValueError(
        #         f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
        #     )
        # if (
        #     rope_scaling_factor is None
        #     or not isinstance(rope_scaling_factor, float)
        #     or rope_scaling_factor <= 1.0
        # ):
        #     raise ValueError(
        #         f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}"
        #     )
