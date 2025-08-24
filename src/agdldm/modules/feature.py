from typing import List, Sequence, Optional, Tuple
import json
import torch
from torch import nn, Tensor


class FeatureSetProjector(nn.Module):
    """
    Projects feature matrices into overlapping feature set slices and reconstructs them.

    This module handles the projection of input feature matrices into multiple overlapping
    feature subsets and provides reconstruction capabilities. It's useful for scenarios
    where features need to be processed in different overlapping groups while maintaining
    the ability to reconstruct the original feature space.

    Parameters
    ----------
    num_features : int
        Total number of features (F) in the input feature matrix.
    feature_set_indices : Sequence[Sequence[int]]
        List of feature sets, where each set contains zero-based column indices (0..F-1)
        defining which features belong to that set. Features can appear in multiple sets
        (overlapping is allowed).
    merge : {"mean", "first", None}, optional, default="mean"
        Strategy for merging overlapping features during reconstruction:

        - "mean": Average values from all sets containing the feature
        - "first": Use value from the first set containing the feature
        - None: No merging, return raw projections (useful for debugging)

    Attributes
    ----------
    num_features : int
        Total number of features in the input matrix.
    num_presets : int
        Number of feature sets (presets).
    p_size : List[int]
        List containing the size of each feature set.
    feature_multiplicity : Tensor
        Count of how many times each feature appears across all sets.

    Methods
    -------
    forward(x)
        Projects input tensor into feature set slices.
    inverse(parts)
        Reconstructs original feature space from slices using merge strategy.
    concat(parts)
        Concatenates feature set slices along feature dimension.
    split(z)
        Splits concatenated tensor back into feature set slices.

    Examples
    --------
    >>> batch_size, num_features = 100, 10
    >>> X = torch.randn(batch_size, num_features)
    >>>
    >>> # Define overlapping feature sets
    >>> feature_sets = [[1, 3, 5, 7, 9], [2, 4, 6, 8], [0, 2, 8]]
    >>>
    >>> # Create projector
    >>> projector = FeatureSetProjector(num_features, feature_sets, merge="mean")
    >>>
    >>> # Project into slices
    >>> parts = projector(X)  # List of tensors with shapes (batch_size, set_size)
    >>>
    >>> # Reconstruct original space
    >>> X_reconstructed = projector.inverse(parts)  # Shape: (batch_size, num_features)
    """

    def __init__(
        self,
        num_features: int,
        feature_set_indices: Sequence[Sequence[int]],
        merge: Optional[str] = "mean",
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_presets = len(feature_set_indices)

        # check feature_set_indices includes all features
        if not set(range(self.num_features)).issubset(
            set(sum(feature_set_indices, []))
        ):
            raise ValueError(f"feature_set_indices must include all features")

        if merge not in (None, "mean", "first"):
            raise ValueError("merge must be one of {None,'mean','first'} (no 'sum').")
        self.merge = merge

        # Store indices as buffers (LongTensor) so .to(device) propagates
        idx_tensors: List[Tensor] = []
        for i, fs in enumerate(feature_set_indices):
            idx = torch.as_tensor(fs, dtype=torch.long)
            if idx.ndim != 1:
                raise ValueError(f"feature set {i} must be 1-D indices.")
            if (idx < 0).any() or (idx >= self.num_features).any():
                bad = idx[(idx < 0) | (idx >= self.num_features)]
                raise IndexError(
                    f"feature set {i} has out-of-range indices, e.g., {bad[:5]}"
                )
            idx_tensors.append(idx)
            self.register_buffer(f"idx_{i}", idx, persistent=True)
        self._idx_list = tuple(idx_tensors)  # for type hints
        self.p_size = [len(fs) for fs in feature_set_indices]

        # Precompute multiplicity per feature for "mean" (used in inverse)
        counts = torch.zeros(self.num_features, dtype=torch.int32)
        for idx in idx_tensors:
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
        self.register_buffer("feature_multiplicity", counts, persistent=True)

    @classmethod
    def from_feature_sets(
        cls,
        feature_list_file: str,
        feature_presets_file: str,
        merge: Optional[str] = "mean",
    ):
        with open(feature_list_file, "r") as f:
            feature_list = json.load(f)
        with open(feature_presets_file, "r") as f:
            feature_presets = json.load(f)
        # feature_presets is a dictionary of {preset_name: [feature_ids]} as string type
        # feature_list is a list of feature_ids as string type
        # convert feature_list to a dictionary of {feature_id: index}
        feature_list_dict = {feature_id: i for i, feature_id in enumerate(feature_list)}
        # convert feature_presets to a dictionary of {preset_name: [feature_ids]} as integer type
        feature_presets_dict = {
            preset_name: [feature_list_dict[feature_id] for feature_id in feature_ids]
            for preset_name, feature_ids in feature_presets.items()
        }
        feature_presets = [
            feature_presets_dict[preset_name] for preset_name in feature_presets_dict
        ]
        return cls(len(feature_list), feature_presets, merge=merge)

    def forward(self, X: Tensor) -> List[Tensor]:
        """
        Project X -> parts
        X: (batch_size, num_features) tensor
        returns: list of length num_presets with shapes (batch_size, p_size[i])
        """
        if X.dim() != 2 or X.size(1) != self.num_features:
            raise ValueError(
                f"X must be ({X.size(0)}, {self.num_features}) but got {X.shape}"
            )
        parts: List[Tensor] = []
        for i in range(self.num_presets):
            idx: Tensor = getattr(self, f"idx_{i}")
            parts.append(X.index_select(dim=1, index=idx))
        return parts

    def inverse(self, parts: Sequence[Tensor]) -> Tensor:
        """
        Reconstruct (batch_size, num_features) from list of parts using merge strategy.
        If weighted, uses weighted mean of overlaps.
        """
        if len(parts) != self.num_presets:
            raise ValueError(f"Expected {self.num_presets} parts, got {len(parts)}")
        batch_size = int(parts[0].size(0))

        device = parts[0].device
        dtype = parts[0].dtype

        for i, P in enumerate(parts):
            if P.dim() != 2 or P.size(0) != batch_size:
                raise ValueError(
                    f"part {i} must be ({batch_size}, {self.p_size[i]}) but got {P.shape}"
                )
            if P.device != device or P.dtype != dtype:
                raise ValueError("all parts must share the same device/dtype")

        X_recon = torch.zeros(
            (batch_size, self.num_features), device=device, dtype=dtype
        )

        if self.merge in (None, "mean"):
            cat_idx = torch.cat(
                [getattr(self, f"idx_{i}") for i in range(self.num_presets)], dim=0
            )
            cat_parts = torch.cat(parts, dim=1)

            X_recon.index_add_(1, cat_idx, cat_parts)

            if self.merge == "mean":
                denom = (
                    self.feature_multiplicity.to(dtype=dtype).clamp_min(1).unsqueeze(0)
                )
                if torch.is_floating_point(X_recon):
                    eps = torch.finfo(dtype).eps
                    denom = denom.clamp_min(eps)
                X_recon = X_recon / denom

            return X_recon

        elif self.merge == "first":
            filled = torch.zeros((self.num_features,), device=device, dtype=torch.bool)

            for i in range(self.num_presets):
                idx: Tensor = getattr(self, f"idx_{i}")  # (m_i,)
                P: Tensor = parts[i]  # (batch_size, m_i)
                mask = ~filled[idx]  # (m_i,)
                if mask.any():
                    cols = idx[mask]
                    X_recon[:, cols] = P[:, mask]
                    filled[cols] = True
            return X_recon
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge}")

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, num_presets={self.num_presets}, merge={self.merge}"


if __name__ == "__main__":
    import torch

    batch_size, F_dim = 100, 10
    X = torch.randn(batch_size, F_dim)

    # Example overlapping feature sets
    fs1 = [1, 3, 5, 7, 9]
    fs2 = [2, 4, 6, 8]
    fs3 = [0, 2, 8]
    feature_sets = [fs1, fs2, fs3]

    proj = FeatureSetProjector(F_dim, feature_sets, merge="mean")  # or "first"
    parts = proj(X)  # forward -> list [(batch_size, m_i)]
    # check parts are correct
    for i, P in enumerate(parts):
        assert torch.allclose(X[:, feature_sets[i]], P)

    X_rec_mean = proj.inverse(parts)  # (batch_size, F)

    assert torch.allclose(X, X_rec_mean)
    print("Test passed")
