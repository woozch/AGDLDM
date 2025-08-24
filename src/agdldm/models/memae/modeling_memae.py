from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from agdldm.modules import FeatureSetProjector, MemoryUnit, FeedForwardNet
from agdldm.models.memae.configuration_memae import MemAEConfig


@dataclass
class MemAEOutput(ModelOutput):
    """Output class for MemAE model."""

    loss: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    att_entropy_loss: Optional[torch.FloatTensor] = None
    att_l1_loss: Optional[torch.FloatTensor] = None
    z: Optional[torch.FloatTensor] = None
    att: Optional[torch.FloatTensor] = None
    x: Optional[torch.FloatTensor] = None
    x_recon: Optional[torch.FloatTensor] = None
    parts: Optional[List[torch.FloatTensor]] = None
    parts_rec: Optional[List[torch.FloatTensor]] = None


class MemAEModel(PreTrainedModel):
    config_class = MemAEConfig
    base_model_prefix = "encoders"
    main_input_name = "input_values"

    def __init__(self, config: MemAEConfig):
        super().__init__(config)
        if config.feature_list_file is None or config.feature_presets_file is None:
            raise ValueError(
                "MemAEConfig.feature_list_file and MemAEConfig.feature_presets_file must be provided."
            )
        self.projector = FeatureSetProjector.from_feature_sets(
            feature_list_file=config.feature_list_file,
            feature_presets_file=config.feature_presets_file,
            merge=config.projector_merge,
        )
        self.num_sets = len(self.projector.p_size)

        # check if at least one loss weight is non-zero and positive
        if (
            self.config.mse_weight <= 0
            and self.config.mse_per_set_weight <= 0
            and self.config.att_entropy_weight <= 0
            and self.config.att_l1_weight <= 0
        ):
            raise ValueError("At least one loss weight must be non-zero and positive.")

        # Per-set encoders: (B, m_i) -> (B, latent_dim)
        self.encoders = nn.ModuleList(
            [
                FeedForwardNet(
                    input_dim=m_i,
                    hidden_dims=config.encoder_hidden_dims,
                    output_dim=config.latent_dim,
                    dropout_rate=config.encoder_dropout_rate,
                    expansion_factor=config.encoder_expansion_factor,
                    layer_type=config.encoder_layer_type,
                    activation_type=config.encoder_activation_type,
                    norm_type=config.encoder_norm_type,
                    use_residual=config.encoder_use_residual,
                    bias=config.encoder_bias,
                )
                for m_i in self.projector.p_size
            ]
        )

        # Shared memory over latent_dim channels, treating set idx as spatial
        if config.use_shared_memory:
            self.memory = MemoryUnit(
                mem_dim=config.mem_dim,
                fea_dim=config.latent_dim,
                shrink_thres=config.shrink_thres,
            )
        else:
            self.memory = nn.ModuleList(
                [
                    MemoryUnit(
                        mem_dim=config.mem_dim,
                        fea_dim=config.latent_dim,
                        shrink_thres=config.shrink_thres,
                    )
                    for _ in range(self.num_sets)
                ]
            )

        # Per-set decoders: (B, latent_dim) -> (B, m_i)
        self.decoders = nn.ModuleList(
            [
                FeedForwardNet(
                    input_dim=config.latent_dim,
                    hidden_dims=config.decoder_hidden_dims,
                    output_dim=m_i,
                    dropout_rate=config.decoder_dropout_rate,
                    expansion_factor=config.decoder_expansion_factor,
                    layer_type=config.decoder_layer_type,
                    activation_type=config.decoder_activation_type,
                    norm_type=config.decoder_norm_type,
                    use_residual=config.decoder_use_residual,
                    bias=config.decoder_bias,
                )
                for m_i in self.projector.p_size
            ]
        )

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _encode_sets(self, parts: List[Tensor]) -> Tensor:
        """Encode list of (B, m_i) -> stacked (B, latent_dim, num_sets)."""
        z_list = [
            enc(P) for enc, P in zip(self.encoders, parts)
        ]  # each (B, latent_dim)
        z = torch.stack(z_list, dim=2)  # (B, latent_dim, num_sets)
        return z

    def _memory(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.config.use_shared_memory:
            out = self.memory(
                z
            )  # output: (B, latent_dim, num_sets), att: (B, mem_dim, num_sets)
            return out["output"], out["att"]
        else:
            out = [
                mem(z[:, :, i]) for mem, i in zip(self.memory, range(self.num_sets))
            ]  # output: (B, latent_dim, num_sets), att: (B, mem_dim, num_sets)
            return out["output"], out["att"]

    def _decode_sets(self, z: Tensor) -> List[Tensor]:
        """Decode (B, latent_dim, num_sets) -> list of (B, m_i)."""
        parts_rec: List[Tensor] = []
        for i, dec in enumerate(self.decoders):
            parts_rec.append(dec(z[:, :, i]))
        return parts_rec

    def _attention_entropy(self, att: Tensor, eps: float = 1e-12) -> Tensor:
        # att: (B, M, P) -> entropy across M per (B,P)
        a = att.clamp_min(eps)
        H = -(a * a.log()).sum(dim=1)  # (B, P)
        return H.mean()

    def _attention_l1(self, att: Tensor) -> Tensor:
        return att.abs().mean()

    def forward(
        self,
        input_values: torch.FloatTensor,  # (B, F)
        labels: Optional[
            torch.FloatTensor
        ] = None,  # optional reconstruction target; defaults to input
        return_dict: bool = True,
    ) -> MemAEOutput | Tuple[torch.Tensor, ...]:
        X = input_values
        parts = self.projector(X)  # list of (B, m_i)
        p = self._encode_sets(parts)  # (B, latent_dim, P)
        z, att = self._memory(p)  # (B, latent_dim, P), (B, M, P)
        parts_rec = self._decode_sets(z)  # list of (B, m_i)
        X_rec = self.projector.inverse(parts_rec)  # (B, F)

        loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)

        if self.config.mse_weight > 0:
            mse = F.mse_loss(X_rec, X, reduction="mean")
            loss += self.config.mse_weight * mse
        if self.config.mse_per_set_weight > 0:
            mse_per_set = [
                F.mse_loss(parts_rec[i], parts[i], reduction="mean")
                for i in range(self.num_sets)
            ]
            loss += self.config.mse_per_set_weight * torch.stack(mse_per_set).mean()
        if self.config.att_entropy_weight > 0:
            att_entropy = self._attention_entropy(att)
            loss += self.config.att_entropy_weight * att_entropy
        else:
            att_entropy = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        if self.config.att_l1_weight > 0:
            att_l1 = self._attention_l1(att)
            loss += self.config.att_l1_weight * att_l1
        else:
            att_l1 = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        if not return_dict:
            return loss, X_rec, z, att, att_entropy, att_l1, parts, parts_rec

        return MemAEOutput(
            loss=loss,
            recon_loss=mse,
            att_entropy_loss=att_entropy,
            att_l1_loss=att_l1,
            z=z,
            att=att,
            x=X,
            x_recon=X_rec,
            parts=parts,
            parts_rec=parts_rec,
        )

    # Convenience methods
    @torch.no_grad()
    def reconstruction_error(self, input_values: Tensor, metric: str = "mse") -> Tensor:
        out = self.forward(input_values, return_dict=True)
        if metric == "mse":
            err = ((out.recon - input_values) ** 2).mean(dim=1)  # per-sample MSE
        elif metric == "mae":
            err = (out.recon - input_values).abs().mean(dim=1)
        else:
            raise ValueError("metric must be one of {'mse','mae'}")
        return err


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    import tempfile
    import os
    import json

    torch.manual_seed(0)

    # Example presets
    input_dim = 20
    fs1 = list(map(str, [0, 2, 4, 6, 8]))
    fs2 = list(map(str, [1, 3, 5, 7, 9]))
    fs3 = list(map(str, [5, 6, 7, 8, 9, 10, 11]))
    fs4 = list(map(str, [12, 13, 14, 15, 16, 17, 18, 19]))
    feature_list = sorted(list(set(fs1 + fs2 + fs3 + fs4)))
    feature_presets = {
        "fs1": fs1,
        "fs2": fs2,
        "fs3": fs3,
        "fs4": fs4,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        feature_list_file = os.path.join(tmpdir, "feature_list.json")
        feature_presets_file = os.path.join(tmpdir, "feature_presets.json")
        with open(feature_list_file, "w") as f:
            json.dump(feature_list, f)
        with open(feature_presets_file, "w") as f:
            json.dump(feature_presets, f)
        import pdb

        pdb.set_trace()
        cfg = MemAEConfig(
            input_dim=input_dim,
            latent_dim=16,
            feature_list_file=feature_list_file,
            feature_presets_file=feature_presets_file,
            # encoder
            encoder_hidden_dims=[64],
            encoder_dropout_rate=0.1,
            encoder_expansion_factor=4,
            encoder_layer_type="mlp",
            encoder_activation_type="gelu",
            encoder_norm_type="post",
            encoder_use_residual=True,
            encoder_bias=True,
            # decoder
            decoder_hidden_dims=[64],
            decoder_dropout_rate=0.1,
            decoder_expansion_factor=4,
            decoder_layer_type="mlp",
            decoder_activation_type="gelu",
            decoder_norm_type="post",
            decoder_use_residual=True,
            decoder_bias=True,
            # memory
            mem_dim=32,
            shrink_thres=0.0025,
            # loss
            mse_weight=1.0,
            mse_per_set_weight=0.5,
            att_entropy_weight=0.001,
            att_l1_weight=0.001,
        )

        model = MemAEModel(cfg)
        X = torch.randn(8, input_dim)
        output = model(X)
        print("loss:", float(output.loss))
        print("recon_loss:", float(output.recon_loss))
        print("att_entropy_loss:", float(output.att_entropy_loss))
        print("att_l1_loss:", float(output.att_l1_loss))
        print("z:", tuple(output.z.shape))
        print("att:", tuple(output.att.shape))
        print("x_recon:", tuple(output.x_recon.shape))
        print("parts:", [tuple(p.shape) for p in output.parts])
        print("parts_rec:", [tuple(p.shape) for p in output.parts_rec])
