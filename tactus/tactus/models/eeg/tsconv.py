"""NICE-style shallow temporal-spatial convolutional EEG encoder (primary baseline).

Topology (Song et al., *NICE*, ICLR 2024; itself a ShallowConvNet/EEGNet descendant)::

    (B, 64, T) -> unsqueeze -> (B, 1, 64, T)
      temporal Conv2d(1, F, (1, kt))        # kt = 25 samples = 125 ms at 200 Hz
      spatial  Conv2d(F, F, (64, 1))        # collapses the 64 electrodes
      BatchNorm2d -> ELU
      AvgPool  (temporal smoothing / decimation)
      Dropout -> flatten                    # (B, F * n_bins)
      ResidualProjection -> (B, D), L2-normalized

Two deliberate departures from the reference implementation, both to make the
trunk usable by :class:`~tactus.models.heads.TimeWindowHeads`:

* ``temporal_padding="same"`` keeps the time axis at ``T``, and
* ``pool_mode="adaptive"`` pools to a fixed ``pool_out`` bins.

Together these make the flattened width independent of ``T``, so the same trunk
runs on the 0-150 / 150-350 / 350-600 ms sub-windows (30 / 40 / 50 samples) as well
as on the full 120-sample epoch.  A fixed 125 ms kernel followed by a fixed
51-sample pool -- the literal NICE configuration -- cannot consume a 30-sample
window at all.  Set ``temporal_padding="valid", pool_mode="fixed"`` to reproduce the
reference behaviour exactly on the full epoch.

Parameter count at defaults (F=40, kt=25, pool_out=16, D=256): ~0.33 M.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..heads import ResidualProjection
from .base import EEGEncoder, count_parameters, register_eeg_encoder

__all__ = ["TemporalSpatialConv", "TSConvEncoder"]


class TemporalSpatialConv(nn.Module):
    """The shared shallow temporal -> spatial convolutional front end.

    Parameters
    ----------
    n_channels:
        Number of electrodes (64).
    n_filters:
        Temporal filter bank size ``F``.
    temporal_kernel:
        Temporal kernel length in samples (25 = 125 ms at 200 Hz).
    temporal_padding:
        ``"same"`` (default, keeps ``T``) or ``"valid"`` (NICE-exact).
    spatial_depthwise:
        ``False`` (default, NICE) uses a full ``Conv2d(F, F, (C, 1))`` that mixes
        electrodes *and* temporal filters.  ``True`` uses EEGNet's depthwise
        ``groups=F`` variant (each temporal filter gets its own spatial filter(s));
        far fewer parameters and often better on small EEG datasets.
    depth_multiplier:
        Spatial filters per temporal filter, depthwise mode only.
    pool_mode:
        ``"adaptive"`` (default) pools to ``pool_out`` bins; ``"fixed"`` uses a
        ``pool_kernel``/``pool_stride`` average pool.
    pool_out, pool_kernel, pool_stride:
        Pooling geometry for the respective modes.
    activation:
        ``"elu"`` (NICE) or ``"square_log"`` (ShallowConvNet's ``log(mean(x**2))``
        power pipeline; more classical for oscillatory features).
    dropout:
        Dropout applied after pooling.
    norm:
        ``"batch"`` (default) or ``"instance"``.  Instance norm is per-trial and
        therefore immune to batch composition, which matters when a batch is a
        stratified condition sample rather than an i.i.d. draw.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        temporal_padding: str = "same",
        spatial_depthwise: bool = False,
        depth_multiplier: int = 1,
        pool_mode: str = "adaptive",
        pool_out: int = 16,
        pool_kernel: int = 17,
        pool_stride: int = 5,
        activation: str = "elu",
        dropout: float = 0.5,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if temporal_padding not in ("same", "valid"):
            raise ValueError(f"temporal_padding must be 'same' or 'valid', got {temporal_padding!r}")
        if pool_mode not in ("adaptive", "fixed"):
            raise ValueError(f"pool_mode must be 'adaptive' or 'fixed', got {pool_mode!r}")
        if activation not in ("elu", "square_log"):
            raise ValueError(f"activation must be 'elu' or 'square_log', got {activation!r}")
        self.n_channels = int(n_channels)
        self.n_filters = int(n_filters)
        self.temporal_kernel = int(temporal_kernel)
        self.temporal_padding = temporal_padding
        self.pool_mode = pool_mode
        self.pool_out = int(pool_out)
        self.pool_kernel = int(pool_kernel)
        self.pool_stride = int(pool_stride)
        self.activation = activation

        pad: Any = "same" if temporal_padding == "same" else 0
        temporal = nn.Conv2d(1, self.n_filters, (1, self.temporal_kernel), padding=pad, bias=False)

        if spatial_depthwise:
            out_ch = self.n_filters * int(depth_multiplier)
            spatial = nn.Conv2d(
                self.n_filters, out_ch, (self.n_channels, 1), groups=self.n_filters, bias=False
            )
        else:
            out_ch = self.n_filters
            spatial = nn.Conv2d(self.n_filters, out_ch, (self.n_channels, 1), bias=False)
        self.out_channels = out_ch

        if norm == "batch":
            norm_layer: nn.Module = nn.BatchNorm2d(out_ch)
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise ValueError(f"norm must be 'batch' or 'instance', got {norm!r}")

        pool: nn.Module
        if pool_mode == "adaptive":
            pool = nn.AdaptiveAvgPool2d((1, self.pool_out))
        else:
            pool = nn.AvgPool2d((1, self.pool_kernel), stride=(1, self.pool_stride))

        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("temporal", temporal),
                    ("spatial", spatial),
                    ("norm", norm_layer),
                ]
            )
        )
        self.pool = pool
        self.drop = nn.Dropout(float(dropout))

    # ---------------------------------------------------------------------------------

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return torch.nn.functional.elu(x)
        # ShallowConvNet power pipeline; the log is applied after pooling in the
        # original, but pooling squared activations then taking the log is the same
        # ordering as braindecode's implementation.
        return torch.square(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T) -> (B, F * n_bins)`` flattened features."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        elif x.dim() != 4:
            raise ValueError(f"expected (B, C, T) or (B, 1, C, T), got {tuple(x.shape)}")
        if x.shape[2] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[2]}")
        if self.temporal_padding == "valid" and x.shape[3] < self.temporal_kernel:
            raise ValueError(
                f"input has {x.shape[3]} samples but the temporal kernel is "
                f"{self.temporal_kernel}; use temporal_padding='same' for short windows"
            )
        h = self.block(x)  # (B, F, 1, T')
        h = self._activate(h)
        if self.pool_mode == "fixed" and h.shape[-1] < self.pool_kernel:
            raise ValueError(
                f"post-conv length {h.shape[-1]} is shorter than pool_kernel={self.pool_kernel}; "
                "use pool_mode='adaptive' for short windows"
            )
        h = self.pool(h)
        if self.activation == "square_log":
            h = torch.log(torch.clamp(h, min=1e-6))
        h = self.drop(h)
        return h.flatten(1)

    def output_dim(self, n_times: int) -> int:
        """Flattened width for an input of ``n_times`` samples (analytic; no forward pass)."""
        t = int(n_times)
        if self.temporal_padding == "valid":
            t = t - self.temporal_kernel + 1
        if t <= 0:
            raise ValueError(f"n_times={n_times} is too short for kernel {self.temporal_kernel}")
        if self.pool_mode == "adaptive":
            bins = self.pool_out
        else:
            bins = (t - self.pool_kernel) // self.pool_stride + 1
            if bins <= 0:
                raise ValueError(f"n_times={n_times} is too short for the fixed pool")
        return self.out_channels * bins


@register_eeg_encoder("tsconv", "nice", "shallow")
class TSConvEncoder(EEGEncoder):
    """NICE-style shallow temporal-spatial conv encoder.  Primary EEG baseline.

    Parameters
    ----------
    n_filters, temporal_kernel, temporal_padding, spatial_depthwise, depth_multiplier,
    pool_mode, pool_out, pool_kernel, pool_stride, conv_activation, conv_dropout, conv_norm:
        Forwarded to :class:`TemporalSpatialConv`.
    proj_hidden:
        Width of the residual projector.  ``None`` uses ``embed_dim``.
    proj_blocks, proj_dropout:
        Residual projector depth and dropout.
    dropout:
        Convenience: when given, sets both ``conv_dropout`` and ``proj_dropout``.
    **base:
        :class:`~tactus.models.eeg.base.EEGEncoder` arguments -- ``n_channels``,
        ``n_times``, ``embed_dim``, ``n_subjects``, ``subject_cond``,
        ``subject_cond_kwargs``, ...
    """

    #: config spellings used by configs/*.yaml
    config_aliases = {
        "n_filters": ("n_spatial_filters", "n_temporal_filters", "F"),
        "proj_hidden": ("d_model", "d_hidden"),
        "conv_dropout": ("drop",),
    }

    def __init__(
        self,
        *,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        temporal_padding: str = "same",
        spatial_depthwise: bool = False,
        depth_multiplier: int = 1,
        pool_mode: str = "adaptive",
        pool_out: int = 16,
        pool_kernel: int = 17,
        pool_stride: int = 5,
        conv_activation: str = "elu",
        conv_dropout: float = 0.5,
        conv_norm: str = "batch",
        proj_hidden: Optional[int] = None,
        proj_blocks: int = 1,
        proj_dropout: float = 0.5,
        dropout: Optional[float] = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        if dropout is not None:  # single knob for both stages
            conv_dropout = proj_dropout = float(dropout)
        self.proj_hidden = proj_hidden
        self.proj_blocks = int(proj_blocks)
        self.proj_dropout = float(proj_dropout)
        self.trunk = TemporalSpatialConv(
            n_channels=self.n_channels,
            n_filters=n_filters,
            temporal_kernel=temporal_kernel,
            temporal_padding=temporal_padding,
            spatial_depthwise=spatial_depthwise,
            depth_multiplier=depth_multiplier,
            pool_mode=pool_mode,
            pool_out=pool_out,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            activation=conv_activation,
            dropout=conv_dropout,
            norm=conv_norm,
        )
        self.setup_head()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.trunk(x)

    def build_projector(self, in_dim: int) -> nn.Module:  # noqa: D102
        return ResidualProjection(
            in_dim,
            self.embed_dim,
            hidden_dim=self.proj_hidden if self.proj_hidden is not None else self.embed_dim,
            n_blocks=self.proj_blocks,
            dropout=self.proj_dropout,
            norm="layernorm",
            normalize=False,  # the base class L2-normalizes once, at the very end
        )


if __name__ == "__main__":  # pragma: no cover - smoke check for the server-side run
    for cond in ("none", "subject_token", "subject_layer", "sulora"):
        enc = TSConvEncoder(n_times=120, embed_dim=256, subject_cond=cond)
        x = torch.randn(4, 64, 120)
        sid = torch.tensor([1, 2, 3, -1])
        z = enc(x, sid)
        print(
            f"{cond:>14s}  params={count_parameters(enc):>9,d}  out={tuple(z.shape)}  "
            f"norm={z.norm(dim=-1).mean():.4f}  rule={enc.unseen_subject_state()['rule']}"
        )
