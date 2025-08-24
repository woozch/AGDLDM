from typing import List, Literal, Optional, Sequence

from transformers import PretrainedConfig

MEMAE_PRESETS = {
    "tiny": {
        "latent_dim": 16,
        "encoder_hidden_dims": [128, 64],
        "encoder_expansion_factor": 4,
        "encoder_layer_type": "mlp",
        "encoder_activation_type": "gelu",
        "encoder_norm_type": "post",
        "encoder_use_residual": True,
        "encoder_bias": True,
        "decoder_hidden_dims": [64, 128],
        "decoder_expansion_factor": 4,
        "decoder_layer_type": "mlp",
        "decoder_activation_type": "gelu",
        "decoder_norm_type": "post",
        "decoder_use_residual": True,
        "decoder_bias": True,
    },
    "small": {
        "latent_dim": 32,
        "encoder_hidden_dims": [256, 128],
        "encoder_expansion_factor": 4,
        "encoder_layer_type": "mlp",
        "encoder_activation_type": "gelu",
        "encoder_norm_type": "post",
        "encoder_use_residual": True,
        "encoder_bias": True,
        "decoder_hidden_dims": [128, 256],
        "decoder_expansion_factor": 4,
        "decoder_layer_type": "mlp",
        "decoder_activation_type": "gelu",
        "decoder_norm_type": "post",
        "decoder_use_residual": True,
        "decoder_bias": True,
    },
    "medium": {
        "latent_dim": 64,
        "encoder_hidden_dims": [512, 256],
        "encoder_expansion_factor": 4,
        "encoder_layer_type": "mlp",
        "encoder_activation_type": "gelu",
        "encoder_norm_type": "post",
        "encoder_use_residual": True,
        "encoder_bias": True,
        "decoder_hidden_dims": [256, 512],
        "decoder_expansion_factor": 4,
        "decoder_layer_type": "mlp",
        "decoder_activation_type": "gelu",
        "decoder_norm_type": "post",
        "decoder_use_residual": True,
        "decoder_bias": True,
    },
    "large": {
        "latent_dim": 128,
        "encoder_hidden_dims": [1024, 512],
        "encoder_expansion_factor": 4,
        "encoder_layer_type": "mlp",
        "encoder_activation_type": "gelu",
        "encoder_norm_type": "post",
        "encoder_use_residual": True,
        "encoder_bias": True,
        "decoder_hidden_dims": [512, 1024],
        "decoder_expansion_factor": 4,
        "decoder_layer_type": "mlp",
        "decoder_activation_type": "gelu",
        "decoder_norm_type": "post",
        "decoder_use_residual": True,
        "decoder_bias": True,
    },
}


class MemAEConfig(PretrainedConfig):
    model_type = "memae"

    def __init__(
        self,
        input_dim: int = 11944,  # 11944 for cluster preset, 11945 for full set, 13974 for all genes
        latent_dim: int = 64,
        # projector configuration
        projector_merge: Optional[str] = "mean",
        feature_list_file: Optional[str] = None,
        feature_presets_file: Optional[str] = None,
        # encoder configuration
        encoder_hidden_dims: List[int] = [512, 256],
        encoder_dropout_rate: float = 0.1,
        encoder_expansion_factor: int = 4,
        encoder_layer_type: Literal["glu", "linear"] = "glu",
        encoder_activation_type: Literal["swish", "relu", "gelu", "sigmoid"] = "swish",
        encoder_norm_type: Literal["pre", "post", "none"] = "post",
        encoder_use_residual: bool = True,
        encoder_bias: bool = True,
        # decoder
        decoder_hidden_dims: List[int] = [256, 512],
        decoder_dropout_rate: float = 0.1,
        decoder_expansion_factor: int = 4,
        decoder_layer_type: Literal["glu", "linear"] = "glu",
        decoder_activation_type: Literal["swish", "relu", "gelu", "sigmoid"] = "swish",
        decoder_norm_type: Literal["pre", "post", "none"] = "post",
        decoder_use_residual: bool = True,
        decoder_bias: bool = True,
        # Memory configuration
        use_shared_memory: bool = True,  # share memory across feature sets
        shrink_thres: float = 0.0025,
        mem_dim: int = 16,
        # Loss configuration
        mse_weight: float = 1.0,
        mse_per_set_weight: float = 0.0,
        att_entropy_weight: float = 0.0002,
        att_l1_weight: float = 0.0,
        # Initialization
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        # projector configuration
        self.feature_list_file = feature_list_file
        self.feature_presets_file = feature_presets_file
        self.projector_merge = projector_merge

        # encoder configuration
        self.encoder_hidden_dims = list(encoder_hidden_dims)
        self.encoder_dropout_rate = float(encoder_dropout_rate)
        self.encoder_expansion_factor = int(encoder_expansion_factor)
        self.encoder_layer_type = encoder_layer_type
        self.encoder_activation_type = encoder_activation_type
        self.encoder_norm_type = encoder_norm_type
        self.encoder_use_residual = encoder_use_residual
        self.encoder_bias = encoder_bias
        # decoder configuration
        self.decoder_hidden_dims = list(decoder_hidden_dims)
        self.decoder_dropout_rate = float(decoder_dropout_rate)
        self.decoder_expansion_factor = int(decoder_expansion_factor)
        self.decoder_layer_type = decoder_layer_type
        self.decoder_activation_type = decoder_activation_type
        self.decoder_norm_type = decoder_norm_type
        self.decoder_use_residual = decoder_use_residual
        self.decoder_bias = decoder_bias

        # Memory configuration
        self.use_shared_memory = use_shared_memory
        self.shrink_thres = float(shrink_thres)
        self.mem_dim = int(mem_dim)

        # Loss configuration
        self.mse_weight = float(mse_weight)
        self.mse_per_set_weight = float(mse_per_set_weight)
        self.att_entropy_weight = float(att_entropy_weight)
        self.att_l1_weight = float(att_l1_weight)

        # Initialization
        self.initializer_range = float(initializer_range)
