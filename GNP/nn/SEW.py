import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ScaleEquivariantWrapper(nn.Module):
    r"""
    A lightweight wrapper that enforces (approximate) scale equivariance
    for a given core network.

    Given right-hand sides :math:`r \in \mathbb{R}^{n \times B}`, this module

    1. Computes a per-RHS scalar norm :math:`s_b`.
    2. Normalizes each column: :math:`\tilde{r}_b = r_b / s_b`.
    3. Applies the core network on the normalized input.
    4. Rescales the output by :math:`s_b`.

    This mirrors the original `ResGCN` behavior in GNP, where inputs (and
    outputs) are normalized by a column-wise norm, and can be reused with
    any core module that expects inputs of shape ``(n, batch_size)``.
    """

    def __init__(self, core: nn.Module, norm: str = "l2", eps: float = 1e-8):
        super().__init__()
        self.core = core
        self.norm = norm
        self.eps = eps

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Exposes the dtype expected by GNP.

        If the wrapped `core` defines a `.dtype` attribute (as in `ResGCN`),
        that value is returned. Otherwise, the dtype of the first parameter
        of `core` is used.
        """
        if hasattr(self.core, "dtype"):
            return self.core.dtype
        return next(self.core.parameters()).dtype

    def _compute_scale(
        self, r: torch.Tensor, mirror: bool = True
    ) -> torch.Tensor:
        r"""
        Computes a per-column scale for the input.

        Parameters
        ----------
        r : torch.Tensor
            Tensor of shape ``(n, batch_size)``.
        mirror : bool
            If True, additionally divides by ``sqrt(n)`` to match the
            normalization used in `ResGCN`.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size,)`` containing the scale for
            each right-hand side.
        """
        n, _ = r.shape

        if self.norm == "l2":
            s = torch.linalg.vector_norm(r, dim=0)
        elif self.norm == "l1":
            s = torch.linalg.vector_norm(r, ord=1, dim=0)
        else:
            raise ValueError(f"Unsupported norm '{self.norm}'")

        if mirror:
            s = s / np.sqrt(n)

        return s

    def forward(
        self, r: torch.Tensor, *core_args, **core_kwargs
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        r : torch.Tensor
            Right-hand sides of shape ``(n, batch_size)``.
        *core_args, **core_kwargs :
            Additional positional and keyword arguments forwarded to `core`.

        Returns
        -------
        torch.Tensor
            Output of the wrapped core network, rescaled to match the
            original input scale.
        """
        # Compute per-column scale and stabilize for division.
        s = self._compute_scale(r)                 # (batch_size,)
        s_safe = torch.clamp(s, min=self.eps)

        # Normalize input.
        r_tilde = r / s_safe                      # broadcast over n

        # Apply the core network.
        x_core = self.core(r_tilde, *core_args, **core_kwargs)

        # Rescale output.
        x = x_core * s_safe                       # broadcast over n
        return x
