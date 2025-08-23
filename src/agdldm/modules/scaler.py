import json
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
import joblib

SCALER_TYPES = (RobustScaler, StandardScaler, PowerTransformer)


def filter_scaler_by_features(
    scaler: SCALER_TYPES,
    feature_list_or_file: str | list[str] | None = None,
) -> SCALER_TYPES:
    if feature_list_or_file is not None:
        scaler = copy.deepcopy(scaler)
        if isinstance(feature_list_or_file, str):
            with open(feature_list_or_file, "r") as file:
                columns = json.load(file)
            assert isinstance(columns, list)
            columns = [str(col) for col in columns]
        else:
            columns = feature_list_or_file
            assert isinstance(columns, list)
            columns = [str(col) for col in columns]

        # select features from scaler from gene_list_file
        if not hasattr(scaler, "feature_names_in_"):
            warnings.warn(
                "scaler does not have feature_names_in_ attribute\n"
                f"scaler.feature_names_in_ will be replaced with list(range(scaler.n_features_in_))"
            )
            scaler.feature_names_in_ = list(map(str, range(scaler.n_features_in_)))

        # check columns is a subset of scaler.feature_names_in_
        if not set(columns).issubset(set(scaler.feature_names_in_)):
            raise ValueError(
                f"columns must be a subset of scaler.feature_names_in_: {scaler.feature_names_in_}\n"
            )

        column_indices = [
            i for i, col in enumerate(scaler.feature_names_in_) if col in columns
        ]
        scaler.n_features_in_ = len(column_indices)
        scaler.feature_names_in_ = columns

        if isinstance(scaler, RobustScaler):
            scaler.center_ = scaler.center_[column_indices]
            scaler.scale_ = scaler.scale_[column_indices]
        elif isinstance(scaler, StandardScaler):
            scaler.mean_ = scaler.mean_[column_indices]
            scaler.var_ = scaler.var_[column_indices]
            if hasattr(scaler, "scale_"):
                scaler.scale_ = scaler.scale_[column_indices]
            if hasattr(scaler, "n_samples_seen_") and isinstance(
                scaler.n_samples_seen_, np.ndarray
            ):
                scaler.n_samples_seen_ = scaler.n_samples_seen_[column_indices]
        elif isinstance(scaler, PowerTransformer):
            scaler.lambdas_ = scaler.lambdas_[column_indices]
            if hasattr(scaler, "_scaler"):
                if not isinstance(scaler._scaler, StandardScaler):
                    raise ValueError(
                        f"scaler._scaler must be a StandardScaler: {type(scaler._scaler)}"
                    )
                scaler._scaler.n_features_in_ = len(column_indices)
                scaler._scaler.mean_ = scaler._scaler.mean_[column_indices]
                scaler._scaler.var_ = scaler._scaler.var_[column_indices]
                if hasattr(scaler._scaler, "scale_"):
                    scaler._scaler.scale_ = scaler._scaler.scale_[column_indices]
                if hasattr(scaler._scaler, "n_samples_seen_") and isinstance(
                    scaler._scaler.n_samples_seen_, np.ndarray
                ):
                    scaler._scaler.n_samples_seen_ = scaler._scaler.n_samples_seen_[
                        column_indices
                    ]
        else:
            raise ValueError(f"Unsupported scaler type: {type(scaler)}")

    return scaler


def load_scaler(scaler_path: str) -> SCALER_TYPES:
    with open(scaler_path, "rb") as file:
        scaler = joblib.load(file)
    return scaler


def save_scaler(scaler: SCALER_TYPES, scaler_path: str):
    with open(scaler_path, "wb") as file:
        joblib.dump(scaler, file)


class ScalerTorchBase(nn.Module):
    RESIZABLE_BUFFERS: tuple[str, ...] = ()

    @classmethod
    def from_sklearn(
        cls,
        scaler: StandardScaler | RobustScaler,
        feature_list_or_file: str | list[str] | None = None,
    ):
        raise NotImplementedError


class StandardScalerTorch(ScalerTorchBase):
    RESIZABLE_BUFFERS = (
        "bias",
        "weight",
    )

    def __init__(self, n_features: int, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        # initialized with empty buffers
        self.register_buffer("bias", torch.zeros(n_features), persistent=True)
        self.register_buffer("weight", torch.ones(n_features), persistent=True)

    @classmethod
    def from_sklearn(
        cls,
        scaler: StandardScaler | RobustScaler,
        feature_list_or_file: str | list[str] | None = None,
    ):
        scaler = filter_scaler_by_features(scaler, feature_list_or_file)
        new_scaler = cls(
            n_features=scaler.n_features_in_,
            with_mean=scaler.with_mean,
            with_std=scaler.with_std,
        )
        if new_scaler.with_mean:
            new_scaler.register_buffer(
                "bias",
                torch.from_numpy(scaler.mean_).to(dtype=torch.float32),
                persistent=True,
            )
        if new_scaler.with_std:
            new_scaler.register_buffer(
                "weight",
                torch.from_numpy(scaler.scale_).to(dtype=torch.float32),
                persistent=True,
            )

        return new_scaler

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        has_bias = hasattr(self, "bias") and self.bias.numel() > 0
        has_weight = hasattr(self, "weight") and self.weight.numel() > 0

        if not (has_bias or has_weight):
            raise ValueError("X is not transformed any, cannot transform")

        if has_bias or has_weight:
            # Calculate reshape dimensions once
            reshape_dims = [1] * (len(X.shape) - 1) + [-1]
            device = X.device

            if has_bias:
                bias = self.bias.reshape(*reshape_dims).to(device)
                X = X - bias
            if has_weight:
                weight = self.weight.reshape(*reshape_dims).to(device)
                X = X / weight

        return X

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        has_bias = hasattr(self, "bias") and self.bias.numel() > 0
        has_weight = hasattr(self, "weight") and self.weight.numel() > 0

        if not (has_bias or has_weight):
            raise ValueError("X is not transformed any, cannot inverse transform")

        if has_bias or has_weight:
            # Calculate reshape dimensions once
            reshape_dims = [1] * (len(X.shape) - 1) + [-1]
            device = X.device

            if has_weight:
                weight = self.weight.reshape(*reshape_dims).to(device)
                X = X * weight
            if has_bias:
                bias = self.bias.reshape(*reshape_dims).to(device)
                X = X + bias

        return X

    def extra_repr(self):
        extra_repr_args = []
        if self.with_mean is False:
            extra_repr_args.append(f"with_mean={self.with_mean}")
        if self.with_std is False:
            extra_repr_args.append(f"with_std={self.with_std}")
        return f"{', '.join(extra_repr_args)}"


class RobustScalerTorch(ScalerTorchBase):
    RESIZABLE_BUFFERS = (
        "bias",
        "weight",
    )

    def __init__(
        self,
        n_features: int,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        unit_variance: bool = False,
    ):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        # initialized with empty buffers
        self.register_buffer("bias", torch.zeros(n_features), persistent=True)
        self.register_buffer("weight", torch.ones(n_features), persistent=True)

    @classmethod
    def from_sklearn(
        cls,
        scaler: StandardScaler | RobustScaler,
        feature_list_or_file: str | list[str] | None = None,
    ):
        scaler = filter_scaler_by_features(scaler, feature_list_or_file)
        new_scaler = cls(
            n_features=scaler.n_features_in_,
            with_centering=scaler.with_centering,
            with_scaling=scaler.with_scaling,
            quantile_range=scaler.quantile_range,
            unit_variance=scaler.unit_variance,
        )
        if new_scaler.with_centering:
            new_scaler.register_buffer(
                "bias",
                torch.from_numpy(scaler.center_).to(dtype=torch.float32),
                persistent=True,
            )
        if new_scaler.with_scaling:
            new_scaler.register_buffer(
                "weight",
                torch.from_numpy(scaler.scale_).to(dtype=torch.float32),
                persistent=True,
            )
        return new_scaler

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        has_bias = hasattr(self, "bias") and self.bias.numel() > 0
        has_weight = hasattr(self, "weight") and self.weight.numel() > 0

        if not (has_bias or has_weight):
            raise ValueError("X is not transformed any, cannot transform")

        if has_bias or has_weight:
            # Calculate reshape dimensions once
            reshape_dims = [1] * (len(X.shape) - 1) + [-1]
            device = X.device

            if has_bias:
                bias = self.bias.reshape(*reshape_dims).to(device)
                X = X - bias
            if has_weight:
                weight = self.weight.reshape(*reshape_dims).to(device)
                X = X / weight

        return X

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        has_bias = hasattr(self, "bias") and self.bias.numel() > 0
        has_weight = hasattr(self, "weight") and self.weight.numel() > 0

        if not (has_bias or has_weight):
            raise ValueError("X is not transformed any, cannot inverse transform")

        if has_bias or has_weight:
            # Calculate reshape dimensions once
            reshape_dims = [1] * (len(X.shape) - 1) + [-1]
            device = X.device

            if has_weight:
                weight = self.weight.reshape(*reshape_dims).to(device)
                X = X * weight
            if has_bias:
                bias = self.bias.reshape(*reshape_dims).to(device)
                X = X + bias

        return X

    def extra_repr(self):
        extra_repr_args = []
        if self.with_centering is False:
            extra_repr_args.append(f"with_centering={self.with_centering}")
        if self.with_scaling is False:
            extra_repr_args.append(f"with_scaling={self.with_scaling}")
        if self.quantile_range[0] != 25.0 and self.quantile_range[1] != 75.0:
            extra_repr_args.append(f"quantile_range={self.quantile_range}")
        if self.unit_variance is True:
            extra_repr_args.append(f"unit_variance={self.unit_variance}")
        return f"{', '.join(extra_repr_args)}"


class PowerTransformerTorch(ScalerTorchBase):
    RESIZABLE_BUFFERS = ("lambdas",)

    def __init__(
        self, n_features: int, method: str = "yeo-johnson", standardize: bool = False
    ):
        super().__init__()
        self.method = method
        self.standardize = standardize
        self.register_buffer("lambdas", torch.ones(n_features), persistent=True)
        if standardize:
            self.scaler = StandardScalerTorch(
                n_features=n_features, with_mean=True, with_std=True
            )
        else:
            self.scaler = None

        self.eps = np.spacing(1.0)

    @classmethod
    def from_sklearn(cls, scaler: PowerTransformer):
        if scaler.method != "yeo-johnson":
            raise ValueError(f"Unsupported method: {scaler.method}")
        new_scaler = cls(
            n_features=scaler.n_features_in_,
            method=scaler.method,
            standardize=scaler.standardize,
        )
        new_scaler.standardize = scaler.standardize
        new_scaler.register_buffer(
            "lambdas",
            torch.from_numpy(scaler.lambdas_).to(dtype=torch.float32),
            persistent=True,
        )
        if hasattr(scaler, "_scaler") and isinstance(scaler._scaler, StandardScaler):
            new_scaler.scaler = StandardScalerTorch.from_sklearn(scaler._scaler)
        return new_scaler

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lambdas") and self.lambdas.numel() > 0:
            X = self._yeo_johnson_transform(X)
        else:
            raise ValueError("lambdas is not initialized, cannot transform")
        if self.scaler is not None:
            X = self.scaler(X)
        return X

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            X = self.scaler.inverse(X)
        if hasattr(self, "lambdas") and self.lambdas.numel() > 0:
            X = self._yeo_johnson_inverse_transform(X)
        else:
            raise ValueError("lambdas is not initialized, cannot inverse transform")
        return X

    def _yeo_johnson_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """
        # x: (batch_size, n_features)
        # lmbda: (n_features,)
        batch_size, n_features = X.shape
        out = torch.zeros_like(X)

        # Expand lmbda to match x shape for batched operations
        lmbda_expanded = self.lambdas.unsqueeze(0).expand(
            batch_size, n_features
        )  # (batch_size, n_features)

        # Create masks for positive and negative values
        pos_mask = X >= 0  # (batch_size, n_features)
        neg_mask = ~pos_mask

        # Handle positive values (x >= 0)
        pos_x = X[pos_mask]
        pos_lmbda = lmbda_expanded[pos_mask]

        # batched computation for positive values
        lmbda_zero_mask = torch.abs(pos_lmbda) < self.eps

        out[pos_mask] = torch.where(
            lmbda_zero_mask,
            torch.log1p(pos_x),  # When lambda ~= 0
            (torch.pow(pos_x + 1, pos_lmbda) - 1) / pos_lmbda,  # When lambda != 0
        )

        if neg_mask.any():
            neg_x = X[neg_mask]
            neg_lmbda = lmbda_expanded[neg_mask]

            # batched computation for negative values
            lmbda_two_mask = torch.abs(neg_lmbda - 2) < self.eps

            out[neg_mask] = torch.where(
                lmbda_two_mask,
                -torch.log1p(-neg_x),  # When lambda ~= 2
                -(torch.pow(-neg_x + 1, 2 - neg_lmbda) - 1)
                / (2 - neg_lmbda),  # When lambda != 2
            )

        return out

    def _yeo_johnson_inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        """
        # x: (batch_size, n_features)
        # lmbda: (n_features,)

        batch_size, n_features = X.shape
        X_inv = torch.zeros_like(X)

        # Expand lmbda to match x shape for batched operations
        lmbda_expanded = self.lambdas.unsqueeze(0).expand(
            batch_size, n_features
        )  # (batch_size, n_features)

        # Create masks for positive and negative values
        pos_mask = X >= 0  # (batch_size, n_features)
        neg_mask = ~pos_mask

        # Handle positive values (x >= 0)
        pos_x = X[pos_mask]
        pos_lmbda = lmbda_expanded[pos_mask]

        # batched computation for positive values
        lmbda_zero_mask = torch.abs(pos_lmbda) < self.eps

        X_inv[pos_mask] = torch.where(
            lmbda_zero_mask,
            torch.exp(pos_x) - 1,  # When lambda ~= 0
            torch.pow(pos_x * pos_lmbda + 1, 1 / pos_lmbda) - 1,  # When lambda != 0
        )

        # Handle negative values (x < 0)
        if neg_mask.any():
            neg_x = X[neg_mask]
            neg_lmbda = lmbda_expanded[neg_mask]

            # batched computation for negative values
            lmbda_two_mask = torch.abs(neg_lmbda - 2) > self.eps

            X_inv[neg_mask] = torch.where(
                lmbda_two_mask,
                1
                - torch.pow(
                    -(2 - neg_lmbda) * neg_x + 1, 1 / (2 - neg_lmbda)
                ),  # When lambda ~= 2
                1 - torch.exp(-neg_x),  # When lambda != 2
            )

        return X_inv

    def extra_repr(self):
        extra_repr_args = []
        if self.method != "yeo-johnson":
            extra_repr_args.append(f"method={self.method}")
        if self.standardize is False:
            extra_repr_args.append(f"standardize={self.standardize}")
        return f"{', '.join(extra_repr_args)}"


def convert_scaler_to_torch_module(scaler: SCALER_TYPES) -> nn.Module:
    if isinstance(scaler, StandardScaler):
        return StandardScalerTorch.from_sklearn(scaler)
    elif isinstance(scaler, RobustScaler):
        return RobustScalerTorch.from_sklearn(scaler)
    elif isinstance(scaler, PowerTransformer):
        return PowerTransformerTorch.from_sklearn(scaler)
    else:
        raise ValueError(f"Unsupported scaler type: {type(scaler)}")


if __name__ == "__main__":
    # from agdldm.utils.scaler import convert_scaler_to_torch_module
    import numpy as np
    import torch
    from functools import partial
    from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

    X = (
        np.random.randn(100, 2)
        + np.random.randn(100, 2) * 2
        - np.random.randn(100, 2) * 0.3
    )
    X_torch = torch.from_numpy(X)
    for scaler_class in [
        StandardScaler,
        partial(StandardScaler, with_std=False),
        partial(RobustScaler, unit_variance=True),
        partial(
            RobustScaler, with_centering=True, with_scaling=False, unit_variance=False
        ),
        partial(PowerTransformer, method="yeo-johnson"),
        partial(PowerTransformer, method="yeo-johnson", standardize=False),
    ]:
        scaler = scaler_class()
        scaler.fit(X)
        # scikit-learn
        X_transformed = scaler.transform(X)
        X_recon = scaler.inverse_transform(X_transformed)

        # torch
        scaler_torch = convert_scaler_to_torch_module(scaler)
        X_transformed_torch = scaler_torch(X_torch)
        X_recon_torch = scaler_torch.inverse(X_transformed_torch)
        if (
            np.isclose(X_transformed, X_transformed_torch.numpy()).all()
            and np.isclose(X_recon, X_recon_torch.numpy()).all()
            and np.isclose(X, X_recon).all()
            and np.isclose(X, X_recon_torch.numpy()).all()
        ):
            print(scaler.__repr__(), scaler_torch.__repr__(), "passed")
        else:
            print(
                scaler.__repr__(),
                scaler_torch.__repr__(),
                "failed",
                np.isclose(X_transformed, X_transformed_torch.numpy()).all(),
                np.isclose(X_recon, X_recon_torch.numpy()).all(),
                np.isclose(X, X_recon).all(),
                np.isclose(X, X_recon_torch.numpy()).all(),
            )
