"""ATM-style EEG encoder: channel self-attention + temporal-spatial conv + residual MLP.

After Li et al., *ATM* (NeurIPS 2024, ``dongyangli-del/EEG_Image_decode``).  The
reference implementation prepends a single ``nn.TransformerEncoderLayer`` to a
NICE-style conv trunk and follows it with a wide residual projector.

Two attention modes are provided; both are agnostic to the number of time samples,
so the encoder also works inside :class:`~tactus.models.heads.TimeWindowHeads`.

``attn_mode="channel_tokens"`` (default)
    Tokens are the **64 electrodes**, with sinusoidal positional encoding over the
    channel index.  Each electrode is summarised by adaptive-average-pooling its time
    course into ``n_summary_bins`` bins and projecting to ``attn_dim``; the transformer
    contextualises electrodes against each other, and a linear read-out emits a
    data-dependent ``(B, 64, 64)`` channel-mixing matrix ``M`` applied as
    ``x <- x + gamma * M x``.  This is a learned, content-dependent re-referencing,
    and it is the natural stimulus-driven counterpart to the *static per-subject*
    re-referencing performed by
    :class:`~tactus.models.eeg.subject_cond.SubjectLayer`.

``attn_mode="time_tokens"``
    The literal ATM ``EEGAttention``: tokens are the ``T`` time samples, the model
    dimension is the channel count (64), and the positional encoding runs over time.

The channel positional encoding is index-based, not montage-geometry-based: BioSemi
64-channel file order is a fixed permutation, so the encoding gives the attention a
consistent channel identity, nothing more.  It carries no scalp-distance information
and must not be described as spatial.

Parameter count at defaults (attn_dim=256, 2 layers, F=40, proj_hidden=1024): ~3.2 M.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ..heads import ResidualProjection
from .base import EEGEncoder, count_parameters, register_eeg_encoder
from .tsconv import TemporalSpatialConv

__all__ = ["sinusoidal_encoding", "ChannelAttention", "TimeAttention", "ATMEncoder"]


def sinusoidal_encoding(n_positions: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Classic sin-cos positional encoding, ``(n_positions, dim)``.

    ``PE[p, 2i] = sin(p / 10000^(2i/dim))``, ``PE[p, 2i+1] = cos(...)``.
    """
    if dim <= 0 or n_positions <= 0:
        raise ValueError("n_positions and dim must be positive")
    pos = torch.arange(n_positions, device=device, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / float(dim))
    pe = torch.zeros(n_positions, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.to(dtype)


def _valid_n_heads(dim: int, requested: int) -> int:
    """Largest divisor of ``dim`` that is <= ``requested`` (``nhead`` must divide ``d_model``)."""
    for h in range(int(requested), 0, -1):
        if dim % h == 0:
            return h
    return 1


class ChannelAttention(nn.Module):
    """Self-attention over electrodes, emitting a data-dependent channel-mixing matrix.

    ``(B, C, T) -> (B, C, T)``, shape-preserving and independent of ``T``.
    """

    def __init__(
        self,
        n_channels: int = 64,
        attn_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        n_summary_bins: int = 32,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.attn_dim = int(attn_dim)
        self.n_summary_bins = int(n_summary_bins)
        heads = _valid_n_heads(self.attn_dim, n_heads)
        self.n_heads = heads

        self.summary = nn.AdaptiveAvgPool1d(self.n_summary_bins)
        self.in_proj = nn.Linear(self.n_summary_bins, self.attn_dim)
        self.register_buffer(
            "pos_encoding", sinusoidal_encoding(self.n_channels, self.attn_dim), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=heads,
            dim_feedforward=self.attn_dim * int(ffn_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.mix_proj = nn.Linear(self.attn_dim, self.n_channels)
        # zero-init the read-out so training starts from exact identity mixing
        nn.init.zeros_(self.mix_proj.weight)
        nn.init.zeros_(self.mix_proj.bias)
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.dim() != 3 or x.shape[1] != self.n_channels:
            raise ValueError(f"expected (B, {self.n_channels}, T), got {tuple(x.shape)}")
        s = self.summary(x)  # (B, C, bins)
        tok = self.in_proj(s) + self.pos_encoding.to(dtype=s.dtype).unsqueeze(0)  # (B, C, attn_dim)
        ctx = self.encoder(tok)  # (B, C, attn_dim)
        mix = self.mix_proj(ctx)  # (B, C, C)
        return x + self.gamma * torch.bmm(mix, x)

    def attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """The channel-mixing matrix ``M`` for this batch, ``(B, C, C)``.

        Exposed for the channel-contribution analyses (posterior vs anterior
        decomposition of the orientation axis): ``M`` says which electrodes each
        output channel draws from.
        """
        s = self.summary(x)
        tok = self.in_proj(s) + self.pos_encoding.to(dtype=s.dtype).unsqueeze(0)
        return self.mix_proj(self.encoder(tok))


class TimeAttention(nn.Module):
    """The literal ATM ``EEGAttention``: attention over time, ``d_model = n_channels``.

    ``(B, C, T) -> (B, C, T)``.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_heads: int = 8,
        n_layers: int = 1,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_len: int = 4096,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        heads = _valid_n_heads(self.n_channels, n_heads)
        self.n_heads = heads
        self.register_buffer(
            "pos_encoding", sinusoidal_encoding(int(max_len), self.n_channels), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.n_channels,
            nhead=heads,
            dim_feedforward=self.n_channels * int(ffn_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.dim() != 3 or x.shape[1] != self.n_channels:
            raise ValueError(f"expected (B, {self.n_channels}, T), got {tuple(x.shape)}")
        t = x.shape[2]
        if t > self.pos_encoding.shape[0]:
            raise ValueError(f"T={t} exceeds the positional-encoding table ({self.pos_encoding.shape[0]})")
        tok = x.transpose(1, 2)  # (B, T, C)
        tok = tok + self.pos_encoding[:t].to(dtype=tok.dtype).unsqueeze(0)
        out = self.encoder(tok).transpose(1, 2)  # (B, C, T)
        return x + self.gamma * out


@register_eeg_encoder("atm")
class ATMEncoder(EEGEncoder):
    """ATM-style encoder: attention front end -> temporal-spatial conv -> residual MLP.

    Parameters
    ----------
    attn_mode:
        ``"channel_tokens"`` (default) or ``"time_tokens"``; see the module docstring.
    attn_dim, attn_heads, attn_layers, attn_ffn_mult, attn_dropout, n_summary_bins,
    residual_scale_init:
        Attention front-end configuration.  ``attn_dim`` is ignored in
        ``"time_tokens"`` mode, where the model dimension is fixed to ``n_channels``.
    n_filters, temporal_kernel, temporal_padding, spatial_depthwise, pool_mode,
    pool_out, conv_dropout:
        Conv trunk configuration, forwarded to
        :class:`~tactus.models.eeg.tsconv.TemporalSpatialConv`.
    proj_hidden, proj_blocks, proj_dropout:
        Residual projector configuration.  ``proj_hidden=1024`` with one block is
        what puts the model in the ~3 M parameter class.
    **base:
        :class:`~tactus.models.eeg.base.EEGEncoder` arguments.
    """

    #: config spellings used by configs/*.yaml
    config_aliases = {
        "attn_dim": ("d_model",),
        "attn_heads": ("n_heads",),
        "attn_layers": ("depth", "n_layers"),
        "n_filters": ("n_spatial_filters", "n_temporal_filters"),
    }

    def __init__(
        self,
        *,
        attn_mode: str = "channel_tokens",
        attn_dim: int = 256,
        attn_heads: int = 8,
        attn_layers: int = 2,
        attn_ffn_mult: int = 2,
        attn_dropout: float = 0.1,
        n_summary_bins: int = 32,
        residual_scale_init: float = 0.1,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        temporal_padding: str = "same",
        spatial_depthwise: bool = False,
        pool_mode: str = "adaptive",
        pool_out: int = 16,
        conv_dropout: float = 0.5,
        conv_norm: str = "batch",
        proj_hidden: Optional[int] = 1024,
        proj_blocks: int = 1,
        proj_dropout: float = 0.5,
        dropout: Optional[float] = None,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        if dropout is not None:  # single knob for the conv and projector stages
            conv_dropout = proj_dropout = float(dropout)
        if attn_mode not in ("channel_tokens", "time_tokens"):
            raise ValueError(f"attn_mode must be 'channel_tokens' or 'time_tokens', got {attn_mode!r}")
        self.attn_mode = attn_mode
        self.proj_hidden = proj_hidden
        self.proj_blocks = int(proj_blocks)
        self.proj_dropout = float(proj_dropout)

        if attn_mode == "channel_tokens":
            self.attention: nn.Module = ChannelAttention(
                n_channels=self.n_channels,
                attn_dim=attn_dim,
                n_heads=attn_heads,
                n_layers=attn_layers,
                ffn_mult=attn_ffn_mult,
                dropout=attn_dropout,
                n_summary_bins=n_summary_bins,
                residual_scale_init=residual_scale_init,
            )
        else:
            self.attention = TimeAttention(
                n_channels=self.n_channels,
                n_heads=attn_heads,
                n_layers=attn_layers,
                ffn_mult=attn_ffn_mult,
                dropout=attn_dropout,
                residual_scale_init=residual_scale_init,
            )

        self.trunk = TemporalSpatialConv(
            n_channels=self.n_channels,
            n_filters=n_filters,
            temporal_kernel=temporal_kernel,
            temporal_padding=temporal_padding,
            spatial_depthwise=spatial_depthwise,
            pool_mode=pool_mode,
            pool_out=pool_out,
            dropout=conv_dropout,
            norm=conv_norm,
        )
        self.setup_head()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.trunk(self.attention(x))

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
    for mode in ("channel_tokens", "time_tokens"):
        enc = ATMEncoder(n_times=120, embed_dim=256, attn_mode=mode, subject_cond="subject_layer")
        x = torch.randn(4, 64, 120)
        z = enc(x, torch.tensor([1, 2, 3, -1]))
        print(f"{mode:>15s}  params={count_parameters(enc):>9,d}  out={tuple(z.shape)}")
