import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class GLU(nn.Module):
    """Gated Linear Unit with various activation types."""

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: Optional[int] = None,
        activation_type: Literal["swish", "relu", "gelu", "sigmoid"] = "swish",
        bias: bool = True,
        multiple_of: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.activation_type = activation_type

        # Calculate hidden dimension
        hidden_dim = int(2 * intermediate_features / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        hidden_dim = hidden_dim // 2  # Split for gate and value

        factory_kwargs = {"device": device, "dtype": dtype}

        # Gate and value projections
        self.gate = nn.Linear(in_features, hidden_dim, bias=bias, **factory_kwargs)
        self.value = nn.Linear(in_features, hidden_dim, bias=bias, **factory_kwargs)

        # Output projection
        self.proj = nn.Linear(hidden_dim, out_features, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        value = self.value(x)

        # Apply activation to gate
        if self.activation_type == "swish":
            gate = F.silu(gate)
        elif self.activation_type == "relu":
            gate = F.relu(gate)
        elif self.activation_type == "gelu":
            gate = F.gelu(gate)
        elif self.activation_type == "sigmoid":
            gate = torch.sigmoid(gate)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

        # Apply gating
        gated = gate * value

        # Project to output
        return self.proj(gated)


class SwiGLU(GLU):
    """Swish-Gated Linear Unit."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, activation_type="swish", bias=bias)


class ReGLU(GLU):
    """ReLU-Gated Linear Unit."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, activation_type="relu", bias=bias)


class GeGLU(GLU):
    """GELU-Gated Linear Unit."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, activation_type="gelu", bias=bias)


class SiGLU(GLU):
    """Sigmoid-Gated Linear Unit."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(
            in_features, out_features, activation_type="sigmoid", bias=bias
        )
