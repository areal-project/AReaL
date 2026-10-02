# SPDX-License-Identifier: Apache-2.0
"""Fit a centered ridge regression from streamed feature/target batches without storing them.

Model: ``target = features @ weight + bias``. With column means ``x_bar``, ``y_bar`` and
centered scatter matrices ``Sxx = X^T X - n x_bar x_bar^T``, ``Sxy = X^T Y - n x_bar y_bar^T``,
    weight = (Sxx + lambda I)^-1 Sxy,   bias = y_bar - x_bar @ weight,
which is the closed form of arXiv:2608.03893 eq. 4 after centering. Accumulation is float64.
Contract: features ``[n, feature_dim]``, targets ``[n, target_dim]``, ``n >= 1`` per update,
at least two rows in total before solving, ``lambda > 0``.
"""

from dataclasses import dataclass

import torch

MIN_ROWS_TO_SOLVE = 2  # one row makes every centered scatter matrix zero


@dataclass(frozen=True, kw_only=True)
class AffineFit:
    """A solved ridge map with its fit diagnostics."""

    weight: torch.Tensor  # [feature_dim, target_dim], float64
    bias: torch.Tensor  # [target_dim], float64
    r_squared: torch.Tensor  # [target_dim], in-sample, float64
    rows: int
    lambda_: float


class RidgeAccumulator:
    """Accumulate the sufficient statistics of a ridge fit in float64."""

    def __init__(
        self, *, feature_dim: int, target_dim: int, device: torch.device | str = "cpu"
    ):
        if feature_dim < 1 or target_dim < 1:
            raise ValueError(
                f"feature_dim and target_dim must be >= 1, got {feature_dim}, {target_dim}"
            )
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.rows = 0
        double = dict(dtype=torch.float64, device=device)
        self._sum_x = torch.zeros(feature_dim, **double)
        self._sum_y = torch.zeros(target_dim, **double)
        self._xtx = torch.zeros(feature_dim, feature_dim, **double)
        self._xty = torch.zeros(feature_dim, target_dim, **double)
        self._yty_diag = torch.zeros(target_dim, **double)

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Add ``n`` rows; shapes must be ``[n, feature_dim]`` and ``[n, target_dim]``."""
        if (
            features.ndim != 2
            or targets.ndim != 2
            or features.shape[0] != targets.shape[0]
        ):
            raise ValueError(
                f"features {tuple(features.shape)} and targets {tuple(targets.shape)} must be "
                "[n, feature_dim] and [n, target_dim] with the same n"
            )
        if features.shape[1] != self.feature_dim or targets.shape[1] != self.target_dim:
            raise ValueError(
                f"expected feature_dim {self.feature_dim} and target_dim {self.target_dim}, got "
                f"{features.shape[1]} and {targets.shape[1]}"
            )
        if features.shape[0] == 0:
            raise ValueError("update requires at least one row")
        x = features.to(self._xtx)
        y = targets.to(self._xty)
        self._sum_x += x.sum(dim=0)
        self._sum_y += y.sum(dim=0)
        self._xtx += x.T @ x
        self._xty += x.T @ y
        self._yty_diag += (y * y).sum(dim=0)
        self.rows += int(features.shape[0])

    def solve(self, *, lambda_: float) -> AffineFit:
        """Solve the centered ridge system; ``lambda_`` must be positive."""
        if lambda_ <= 0:
            raise ValueError(
                f"lambda_ must be > 0 for a well-posed solve, got {lambda_}"
            )
        if self.rows < MIN_ROWS_TO_SOLVE:
            raise ValueError(
                f"solve requires at least {MIN_ROWS_TO_SOLVE} rows, got {self.rows}"
            )
        rows = float(self.rows)
        x_bar = self._sum_x / rows
        y_bar = self._sum_y / rows
        sxx = self._xtx - rows * torch.outer(x_bar, x_bar)
        sxy = self._xty - rows * torch.outer(x_bar, y_bar)
        syy_diag = self._yty_diag - rows * y_bar * y_bar
        regularized = sxx + lambda_ * torch.eye(
            self.feature_dim, dtype=sxx.dtype, device=sxx.device
        )
        try:
            factor = torch.linalg.cholesky(regularized)
        except RuntimeError as error:
            raise ValueError(
                f"Sxx + {lambda_} I is not positive definite; raise lambda_ or add rows"
            ) from error
        weight = torch.cholesky_solve(sxy, factor)
        bias = y_bar - x_bar @ weight
        residual_diag = (
            syy_diag
            - 2 * (weight * sxy).sum(dim=0)
            + (weight * (sxx @ weight)).sum(dim=0)
        )
        r_squared = torch.where(
            syy_diag > 0, 1.0 - residual_diag / syy_diag, torch.zeros_like(syy_diag)
        )
        return AffineFit(
            weight=weight,
            bias=bias,
            r_squared=r_squared,
            rows=self.rows,
            lambda_=lambda_,
        )
