import torch
import torch.nn as nn
from typing import Literal, Optional, List

from .glu import GLU
from transformers.activations import ACT2FN


class FFNBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: Optional[int] = None,
        dropout_rate: float = 0.1,
        layer_type: Literal["glu", "mlp"] = "glu",
        activation_type: Literal["swish", "relu", "gelu", "sigmoid"] = "swish",
        norm_type: Literal["pre", "post", "none"] = "post",
        use_residual: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.layer_type = layer_type
        self.norm_type = norm_type
        self.use_residual = use_residual

        # Create normalization layer
        if norm_type != "none":
            self.norm = nn.LayerNorm(
                in_features if norm_type == "pre" else out_features
            )
        else:
            self.norm = nn.Identity()

        # Create FeedForward Network layer
        if layer_type == "glu":  # gated linear unit
            self.block = GLU(
                in_features,
                intermediate_features,
                out_features,
                activation_type=activation_type,
                bias=bias,
            )
        elif layer_type == "mlp":  # multi-layer perceptron
            self.block = nn.Sequential(
                nn.Linear(in_features, intermediate_features, bias=bias),
                ACT2FN[activation_type],
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_features, out_features, bias=bias),
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # Create residual connection if needed
        if use_residual and in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            residual = self.skip(x)

            if self.norm_type == "pre":
                # Pre-norm: norm -> encoder -> dropout -> residual
                x = self.norm(x)
                x = self.block(x)
                x = self.dropout(x)
                x = x + residual
            elif self.norm_type == "post":
                # Post-norm: encoder -> dropout -> residual -> norm
                x = self.block(x)
                x = self.dropout(x)
                x = self.norm(x + residual)
            else:  # norm_type == "none"
                # No norm: encoder -> dropout -> residual
                x = self.block(x)
                x = self.dropout(x)
                x = x + residual
        else:
            if self.norm_type == "pre":
                # Pre-norm: norm -> encoder -> dropout -> residual
                x = self.norm(x)
                x = self.block(x)
                x = self.dropout(x)
            elif self.norm_type == "post":
                # Post-norm: encoder -> dropout -> residual -> norm
                x = self.block(x)
                x = self.dropout(x)
                x = self.norm(x)
            else:  # norm_type == "none"
                # No norm: encoder -> dropout -> residual
                x = self.block(x)
                x = self.dropout(x)

        return x


class FeedForwardNet(nn.Module):
    """FeedForward Neural Network with various layer types."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        expansion_factor: int = 4,
        layer_type: Literal["glu", "linear"] = "glu",
        activation_type: Literal["swish", "relu", "gelu", "sigmoid"] = "swish",
        norm_type: Literal["pre", "post", "none"] = "post",
        use_residual: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        # Set output dimension
        if output_dim is None:
            output_dim = hidden_dims[-1]

        # Create layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(
                FFNBlock(
                    prev_dim,
                    prev_dim * expansion_factor,
                    hidden_dim,
                    dropout_rate=dropout_rate,
                    layer_type=layer_type,
                    activation_type=activation_type,
                    norm_type=norm_type,
                    use_residual=use_residual,
                    bias=bias,
                )
            )
            prev_dim = hidden_dim

        # Add output layer if needed
        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
