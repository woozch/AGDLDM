import torch
from torch import nn
import math
from torch.nn import functional as F


@torch.jit.script
def hard_shrink_pos(x: torch.Tensor, lambd: float = 0.0025):
    if lambd <= 0:
        return x
    return torch.where(x > lambd, x, torch.zeros_like(x))


class MemoryUnit(nn.Module):
    """
    Unified MemoryUnit:
      - If input is 2D: (T, C) -> returns {"output": (T, C), "att": (T, M)}
      - If input is 3D/4D/5D: (N, C, *S) ->
          returns {"output": (N, C, *S), "att": (N, M, *S)}
    """

    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025):
        super().__init__()
        assert mem_dim > 0 and fea_dim > 0
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.empty(self.mem_dim, self.fea_dim))
        self.bias = None  # kept for API compatibility; not used
        self.shrink_thres = float(shrink_thres)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)

    def _address_flat(self, x_flat: torch.Tensor):
        """
        x_flat: (T, C)
        returns: out_flat (T, C), att (T, M)
        """
        att = F.linear(x_flat, self.weight)  # (T, M)
        att = F.softmax(att, dim=1)  # (T, M)

        if self.shrink_thres > 0:
            att = hard_shrink_pos(att, self.shrink_thres)
            att_sum = att.sum(dim=1, keepdim=True).clamp_min(1e-12)
            att = att / att_sum

        out_flat = F.linear(att, self.weight.permute(1, 0))  # (T, C)
        return out_flat, att

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            # (T, C)
            if x.shape[1] != self.fea_dim:
                raise ValueError(f"Expected last dim {self.fea_dim}, got {x.shape[1]}")
            out_flat, att = self._address_flat(x)
            return {"output": out_flat, "att": att}

        if x.ndim not in (3, 4, 5):
            raise ValueError(f"Unsupported input dim {x.ndim}. Expected 2/3/4/5D.")

        # (N, C, *S) -> flatten to (T, C)
        N, C = x.shape[0], x.shape[1]
        if C != self.fea_dim:
            raise ValueError(f"Expected channel dim {self.fea_dim}, got {C}")
        spatial_shape = x.shape[2:]
        x_last = x.movedim(1, -1).contiguous()  # (N, *S, C)
        x_flat = x_last.view(-1, C)  # (T, C)

        # address & project
        y_flat, att_flat = self._address_flat(x_flat)  # (T, C), (T, M)

        # reshape back
        y = y_flat.view(N, *spatial_shape, C).movedim(-1, 1).contiguous()  # (N, C, *S)
        att = (
            att_flat.view(N, *spatial_shape, self.mem_dim).movedim(-1, 1).contiguous()
        )  # (N, M, *S)

        return {"output": y, "att": att}

    def extra_repr(self):
        return f"mem_dim={self.mem_dim}, fea_dim={self.fea_dim}, shrink_thres={self.shrink_thres}"


if __name__ == "__main__":
    mem = MemoryUnit(2, 5)
    x = torch.randn(10, 5)
    print(mem(x))
