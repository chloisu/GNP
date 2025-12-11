import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ScaleEquivariantWrapper(nn.Module):
    def __init__(self, core, norm="l2", eps=1e-8):
        super().__init__()
        self.core = core
        self.norm = norm
        self.eps  = eps

    @property
    def dtype(self):
        # ResGCN defines .dtype; fall back to parameter dtype if needed
        if hasattr(self.core, "dtype"):
            return self.core.dtype
        else:
            return next(self.core.parameters()).dtype

    def _compute_scale(self, r: torch.Tensor, mirror=True) -> torch.Tensor:
        """
        r: [n, batch_size]

        Return per-RHS scale s of shape [batch_size].
        This matches the original ResGCN behavior of a
        column-wise norm.
        """
        n, batch_size = r.shape

        if self.norm == "l2":
            s = torch.linalg.vector_norm(r, dim=0)  # [batch_size]
        elif self.norm == "l1":
            s = torch.linalg.vector_norm(r, ord=1, dim=0)
        else:
            raise ValueError(f"Unsupported norm '{self.norm}'")

        if mirror:
            s = s / np.sqrt(n)

        return s

    def forward(self, r: torch.Tensor, *core_args, **core_kwargs) -> torch.Tensor:
        """
        r: [n, batch_size] right-hand sides
        """
        # 1) compute per-column scale
        s = self._compute_scale(r)                  # [batch_size]
        s_safe = torch.clamp(s, min=self.eps)       # avoid div-by-zero

        # 2) normalize input
        r_tilde = r / s_safe                        # broadcasts over n

        # 3) run the original core network on normalized input
        x_core = self.core(r_tilde, *core_args, **core_kwargs)

        # 4) rescale output
        x = x_core * s_safe                         # same broadcast
        return x
