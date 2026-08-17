"""Projection heads shared by the EEG and video towers.

Contract F (repo-level):
    ``VideoProjector(nn.Module): (B, D_vid) -> (B, D)`` L2-normalized.

This module also hosts two pieces of shared plumbing:

* :class:`ResidualProjection` -- the NICE/ATM style ``Linear -> ResidualAdd(GELU,
  Linear, Dropout) -> LayerNorm`` projector reused by every EEG encoder, so that
  the EEG and video towers are projected by structurally identical heads.
* :class:`TimeWindowHeads` -- the time-resolved alignment wrapper.  It runs the
  (optionally shared) EEG trunk on sub-windows of the epoch and gives each window
  its own projection head, which operationalizes BLUEPRINT_v2 contribution 2
  ("分窗投影头产出对齐起始曲线", windows 0-150 / 150-350 / 350-600 ms).

Nothing here touches the loss contract: every head returns L2-normalized
embeddings so ``ContrastiveLoss.forward`` always sees unit vectors.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ._cfg import apply_aliases, filter_kwargs, to_dict, unwrap_section

__all__ = [
    "l2_normalize",
    "ResidualAdd",
    "ResidualProjection",
    "VideoProjector",
    "build_video_projector",
    "VideoSequenceProjector",
    "build_video_sequence_projector",
    "FRAME_POOLINGS",
    "TimeWindowHeads",
    "DEFAULT_TIME_WINDOWS_MS",
    "EPOCH_WINDOW_SPECS",
]


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

#: Time-resolved alignment windows, in milliseconds relative to stimulus onset.
#: Chosen to bracket the companion paper's MVPA onsets (hand orientation ~60 ms,
#: material 110-120 ms, valence onset ~130 ms / peak 300 ms, touch type 165 ms,
#: threat/arousal 230-260 ms), so the alignment-onset curve is directly comparable.
DEFAULT_TIME_WINDOWS_MS: Dict[str, Tuple[float, float]] = {
    "early": (0.0, 150.0),
    "mid": (150.0, 350.0),
    "late": (350.0, 600.0),
}

#: Epoch storage windows from contract B, with the metadata needed to convert
#: milliseconds to sample indices.
EPOCH_WINDOW_SPECS: Dict[str, Dict[str, float]] = {
    "w0600": {"tmin_ms": 0.0, "tmax_ms": 600.0, "sfreq": 200.0, "n_times": 120},
    "wm100_800": {"tmin_ms": -100.0, "tmax_ms": 800.0, "sfreq": 200.0, "n_times": 180},
}


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize ``x`` along ``dim`` with a numerically safe denominator.

    ``F.normalize`` clamps the norm at ``eps`` which yields a vector of norm
    ``|x| / eps`` for near-zero inputs; here we add ``eps`` to the norm instead so
    a zero vector maps to a zero vector rather than exploding.
    """
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class ResidualAdd(nn.Module):
    """``y = x + fn(x)``.  Used to keep the NICE/ATM projector topology explicit."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + self.fn(x)


class ResidualProjection(nn.Module):
    """Flatten-head projector: ``Linear -> n_blocks x ResidualAdd(GELU, Linear, Drop) -> Norm``.

    Parameters
    ----------
    in_dim:
        Input dimensionality (flattened trunk features).
    out_dim:
        Output embedding dimensionality ``D``.
    hidden_dim:
        Width of the residual stack.  ``None`` uses ``out_dim`` (NICE behaviour).
        When ``hidden_dim != out_dim`` a final ``Linear(hidden_dim, out_dim)`` is
        appended.
    n_blocks:
        Number of residual blocks.
    dropout:
        Dropout probability inside each residual block.
    norm:
        ``"layernorm"``, ``"batchnorm"`` or ``None``; applied to the output.
    normalize:
        If ``True`` the output is L2-normalized (the default for embedding heads).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        n_blocks: int = 1,
        dropout: float = 0.5,
        norm: Optional[str] = "layernorm",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim must be positive, got {in_dim}/{out_dim}")
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(out_dim)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = hidden_dim
        self.normalize = bool(normalize)

        layers: list[nn.Module] = [nn.Linear(self.in_dim, hidden_dim)]
        for _ in range(int(n_blocks)):
            layers.append(
                ResidualAdd(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(float(dropout)),
                    )
                )
            )
        if hidden_dim != self.out_dim:
            layers.append(nn.Linear(hidden_dim, self.out_dim))
        if norm == "layernorm":
            layers.append(nn.LayerNorm(self.out_dim))
        elif norm == "batchnorm":
            layers.append(nn.BatchNorm1d(self.out_dim))
        elif norm not in (None, "none"):
            raise ValueError(f"unknown norm {norm!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        z = self.net(x)
        return l2_normalize(z) if self.normalize else z

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, out_dim={self.out_dim}"


# --------------------------------------------------------------------------------------
# video side (contract F)
# --------------------------------------------------------------------------------------


class VideoProjector(nn.Module):
    """Project a *frozen* video embedding into the joint space.

    Contract F: ``(B, D_vid) -> (B, D)``, L2-normalized.

    The frozen video embeddings live in ``data/derived/video_emb/{tag}.npz`` and are
    already L2-normalized (contract C), so this head only has to learn a rotation /
    re-weighting into the EEG-alignable subspace.  Keep it small: with a fixed
    codebook of at most 360 conditions a large head memorizes the codebook.

    Parameters
    ----------
    in_dim:
        ``D_vid`` of the frozen encoder (e.g. 768 for SigLIP2-base, 1024 for V-JEPA2-L).
    out_dim:
        Joint-space dimensionality ``D``.
    hidden_dim:
        If given (and ``n_layers > 1``) the width of the MLP hidden layer.
    n_layers:
        1 -> single ``Linear`` (default, recommended); 2 -> ``Linear-GELU-Dropout-Linear``.
    dropout:
        Dropout applied between MLP layers (ignored when ``n_layers == 1``).
    norm:
        Output normalization layer, see :class:`ResidualProjection`.
    bias:
        Whether the linear layers carry a bias.
    normalize:
        L2-normalize the output.  Only turn this off for diagnostics -- the loss
        contract assumes unit vectors.
    residual_blocks:
        If > 0, insert that many :class:`ResidualAdd` blocks (mirrors the EEG head).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: Optional[int] = None,
        n_layers: int = 1,
        dropout: float = 0.0,
        norm: Optional[str] = None,
        bias: bool = True,
        normalize: bool = True,
        residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim must be positive, got {in_dim}/{out_dim}")
        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.normalize = bool(normalize)

        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim, bias=bias))
            width = self.out_dim
        else:
            hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.out_dim, self.in_dim // 2)
            layers += [
                nn.Linear(self.in_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, self.out_dim, bias=bias),
            ]
            width = self.out_dim
        for _ in range(int(residual_blocks)):
            layers.append(
                ResidualAdd(
                    nn.Sequential(nn.GELU(), nn.Linear(width, width, bias=bias), nn.Dropout(float(dropout)))
                )
            )
        if norm == "layernorm":
            layers.append(nn.LayerNorm(self.out_dim))
        elif norm == "batchnorm":
            layers.append(nn.BatchNorm1d(self.out_dim))
        elif norm not in (None, "none"):
            raise ValueError(f"unknown norm {norm!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """``v``: ``(B, D_vid)`` (or ``(..., D_vid)``) -> ``(B, D)`` L2-normalized."""
        if v.shape[-1] != self.in_dim:
            raise ValueError(f"VideoProjector expected last dim {self.in_dim}, got {tuple(v.shape)}")
        lead = v.shape[:-1]
        z = self.net(v.reshape(-1, self.in_dim))
        z = z.reshape(*lead, self.out_dim)
        return l2_normalize(z) if self.normalize else z

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


#: Accepted spellings for projector options (``canonical -> aliases``).
VIDEO_PROJECTOR_ALIASES: Dict[str, Sequence[str]] = {
    "in_dim": ("d_in", "d_video", "d_vid", "video_dim", "vid_dim", "input_dim"),
    "out_dim": ("d_out", "d_embed", "embed_dim", "dim"),
    "hidden_dim": ("d_hidden", "hidden"),
}

#: Named projector shapes; ``build_video_projector("mlp", ...)`` picks one.
VIDEO_PROJECTOR_KINDS: Dict[str, Dict[str, Any]] = {
    "linear": {"n_layers": 1},
    "mlp": {"n_layers": 2},
    "mlp2": {"n_layers": 2},
    "residual": {"n_layers": 1, "residual_blocks": 1},
}


def build_video_projector(cfg: Any = None, **overrides: Any) -> VideoProjector:
    """Build the frozen-video projection head (contract F).

    Accepts a whole run config with a ``video_projector`` / ``projector`` section, the
    section itself, or a bare kind name (``"linear"``, ``"mlp"``, ``"residual"``), plus
    keyword overrides.  ``in_dim`` must equal ``D_vid`` of the frozen encoder -- read it
    from the cache rather than hard-coding it::

        meta = json.loads(str(np.load(f"{tag}.npz")["meta"].item()))
        proj = build_video_projector(in_dim=meta["embedding_dim"], out_dim=256)
    """
    kind: Optional[str] = None
    if isinstance(cfg, str):
        kind, conf = cfg, {}
    else:
        conf = dict(unwrap_section(to_dict(cfg), "video_projector", "video_head", "projector"))
    conf.update(overrides)
    apply_aliases(conf, VIDEO_PROJECTOR_ALIASES)
    conf.pop("_target_", None)
    params = conf.pop("params", None)
    if params is not None:
        merged = to_dict(params)
        merged.update(conf)
        conf = apply_aliases(merged, VIDEO_PROJECTOR_ALIASES)
    kind = str(conf.pop("name", None) or conf.pop("kind", None) or kind or "linear").lower()

    defaults = dict(VIDEO_PROJECTOR_KINDS.get(kind, {}))
    for key, value in defaults.items():
        conf.setdefault(key, value)
    if "in_dim" not in conf:
        raise KeyError(
            "build_video_projector needs in_dim (= D_vid of the frozen video encoder). "
            "Read it from the cache: "
            "json.loads(str(np.load(path)['meta'].item()))['embedding_dim']"
        )
    kwargs = filter_kwargs(VideoProjector, conf, f"build_video_projector({kind!r})")
    return VideoProjector(**kwargs)


# --------------------------------------------------------------------------------------
# video side, sequence form (contract F-seq)
# --------------------------------------------------------------------------------------
#
# Contract C stores one *pooled* vector per condition, so frame order is destroyed before
# the EEG model sees anything.  ``tactus.models.video.temporal`` writes the same encoder
# output with the pooling left undone -- ``cond_seq (360, T, D)`` -- and the heads below
# consume that: ``(B, T, D_vid) -> (B, D)``, L2-normalized, i.e. the same output contract
# as :class:`VideoProjector`.
#
# Two design decisions that make the arms comparable rather than merely different:
#
# 1. *The projector body is literally* :class:`VideoProjector`.  A pooler reduces
#    ``(B, T, D) -> (B, D)``, the result is L2-normalized (contract C's pooled vector is
#    ``l2(mean_t f_t)``, so the mean pooler's output *is* contract C), and the same MLP
#    runs on top.  ``pool="mean"`` is therefore not "approximately the old behaviour", it
#    is the old behaviour, with byte-identical parameter names under the ``proj.`` prefix.
#
# 2. Every temporal pooler can be *initialised to the mean* (``init_as_mean=True``,
#    default).  At step 0 an attention-pooled arm and a mean-pooled arm emit identical
#    embeddings, so a later difference in the endpoint is attributable to what training
#    found in the temporal axis and not to a lucky initialisation.
#
# The trap this avoids: **content-based attention pooling over frames is permutation
# invariant.**  Softmax-weighted summation of a set is order-blind no matter how many
# parameters it has, which is exactly the failure mode the census found in X-CLIP
# (cos(native, shuffled) = 0.9999).  ``_AttentionFramePool`` therefore adds a *learned
# positional embedding* before scoring; with ``learn_pos=False`` it collapses back to an
# order-blind set-pooler and is only useful as a control.


class _MeanFramePool(nn.Module):
    """``(B, T, D) -> (B, D)`` by the arithmetic mean.  Order-blind by construction."""

    order_sensitive = False

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return v.mean(dim=1)


class _PositionWeightFramePool(nn.Module):
    """Softmax over ``T`` learned per-position scalars: the minimal temporal read-out.

    ``T`` parameters, content-independent.  It cannot express "attend to the frame where
    the hand lands", but it can express "the last third of the clip carries the signal",
    and it reduces to the mean when the logits are equal -- which is where it starts.
    """

    order_sensitive = True

    def __init__(self, n_frames: int, init_as_mean: bool = True) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        # equal logits -> uniform softmax -> exactly the mean
        self.logits = nn.Parameter(torch.zeros(self.n_frames) if init_as_mean else torch.randn(self.n_frames))

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa: D102
        w = torch.softmax(self.logits, dim=0).to(v.dtype)
        return torch.einsum("t,btd->bd", w, v)

    def extra_repr(self) -> str:  # noqa: D102
        return f"n_frames={self.n_frames}"


class _AttentionFramePool(nn.Module):
    """Single-query attention pooling over frames, with learned positions.

    ``score_t = <q, W_k (x_t + p_t)> / sqrt(d_a)``, ``alpha = softmax_t(score)``,
    ``out = sum_t alpha_t * W_v (x_t + p_t)`` (``W_v = I`` by default, so the output stays
    in the frozen encoder's own space and the downstream ``proj`` sees the same geometry
    as the mean arm).

    Without ``p_t`` this module is permutation invariant -- see the note above.
    """

    order_sensitive = True

    def __init__(
        self,
        in_dim: int,
        n_frames: Optional[int] = None,
        attn_dim: Optional[int] = None,
        learn_pos: bool = True,
        value_proj: bool = False,
        init_as_mean: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if learn_pos and not n_frames:
            raise ValueError("_AttentionFramePool with learn_pos=True needs n_frames")
        self.in_dim = int(in_dim)
        self.n_frames = int(n_frames) if n_frames else None
        self.attn_dim = int(attn_dim) if attn_dim else min(self.in_dim, 128)
        self.learn_pos = bool(learn_pos)
        self.pos = nn.Parameter(torch.zeros(self.n_frames, self.in_dim)) if learn_pos else None
        # LayerNorm *before* the key projection.  Frozen encoder embeddings are unit
        # vectors, so their components are ~1/sqrt(D); feeding those straight into
        # ``<q, W_k x> / sqrt(d_a)`` gives scores with s.d. ~0.03, a softmax that is
        # uniform to three decimals, and an "attention" pooler that is the mean whatever
        # its weights say.  Measured: without this, cos(native, shuffled) at random init
        # was 0.99985; with it, 0.95.  It does not touch the value path, so
        # ``init_as_mean`` still reproduces the mean exactly.
        self.pre = nn.LayerNorm(self.in_dim)
        self.key = nn.Linear(self.in_dim, self.attn_dim)
        self.query = nn.Parameter(torch.zeros(self.attn_dim))
        self.value = nn.Linear(self.in_dim, self.in_dim) if value_proj else None
        self.drop = nn.Dropout(float(dropout))
        if init_as_mean:
            # constant scores -> uniform softmax -> exactly the mean, and a value path
            # that is the identity.  Only ``pos`` and ``query`` need to move for the head
            # to become temporal, so gradient descent starts from the current behaviour.
            nn.init.zeros_(self.query)
            if self.value is not None:
                nn.init.eye_(self.value.weight)
                nn.init.zeros_(self.value.bias)
        else:
            # A zero query is a *constant* score vector, hence a uniform softmax, hence the
            # mean -- order-blind however the rest is initialised.  init_as_mean=False must
            # therefore randomise the query, not just the positions.
            nn.init.normal_(self.query, std=1.0)
            if self.pos is not None:
                # frames are unit vectors, so a component is ~1/sqrt(D); match that scale
                nn.init.normal_(self.pos, std=float(self.in_dim) ** -0.5)

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = v
        if self.pos is not None:
            if x.shape[1] != self.pos.shape[0]:
                raise ValueError(
                    f"_AttentionFramePool was built for T={self.pos.shape[0]}, got T={x.shape[1]}"
                )
            x = x + self.pos.to(x.dtype)
        scores = self.key(self.pre(x)) @ self.query.to(x.dtype)  # (B, T)
        alpha = self.drop(torch.softmax(scores / (self.attn_dim ** 0.5), dim=1))
        val = self.value(x) if self.value is not None else x
        return torch.einsum("bt,btd->bd", alpha, val)

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_dim={self.in_dim}, attn_dim={self.attn_dim}, n_frames={self.n_frames}, "
            f"learn_pos={self.learn_pos}"
        )


class _TemporalConvFramePool(nn.Module):
    """Residual temporal conv over the frame axis, then a *non-uniform* temporal pool.

    ``out = sum_t w_t (x_t + f(x)_t)`` with ``f`` a ``Conv1d(k>1) -> GELU -> Conv1d`` stack
    whose last layer is zero-initialised, and ``w = softmax(logits)`` over the ``T``
    positions (uniform at init).  A convolution with ``k > 1`` mixes neighbouring frames,
    so ``f`` is order-sensitive; the residual and the zero-init mean the module *starts*
    at exactly the mean.

    Why ``w`` and not a plain mean.  ``mean_t (conv * x)_t`` is, for a linear convolution,
    ``(sum of the kernel taps) * mean_t x_t`` up to the boundary -- i.e. **averaging over
    time undoes the convolution**, and a conv-then-mean read-out is order-blind except for
    edge effects and whatever the nonlinearity leaks.  Measured on a synthetic sequence
    with a strong planted temporal ramp, conv-then-mean gave
    ``Spearman(RDM_native, RDM_shuffled) = 0.99999``: a "temporal" head that sees nothing.
    A non-uniform ``w`` is what turns the mixed features back into a temporal read-out.
    """

    def __init__(
        self,
        in_dim: int,
        n_frames: Optional[int] = None,
        conv_channels: Optional[int] = None,
        kernel_size: int = 3,
        n_layers: int = 1,
        dropout: float = 0.0,
        init_as_mean: bool = True,
    ) -> None:
        super().__init__()
        if int(kernel_size) < 2:
            raise ValueError(
                f"kernel_size={kernel_size} makes the conv per-frame, hence order-blind; use >= 2"
            )
        self.in_dim = int(in_dim)
        self.kernel_size = int(kernel_size)
        self.n_frames = int(n_frames) if n_frames else None
        self.weights = (
            _PositionWeightFramePool(self.n_frames, init_as_mean=init_as_mean)
            if self.n_frames
            else _MeanFramePool()
        )
        ch = int(conv_channels) if conv_channels else min(self.in_dim, 256)
        pad = self.kernel_size // 2
        layers: list[nn.Module] = []
        d_in = self.in_dim
        for _ in range(max(int(n_layers), 1)):
            layers += [
                nn.Conv1d(d_in, ch, self.kernel_size, padding=pad),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
            d_in = ch
        out_conv = nn.Conv1d(d_in, self.in_dim, 1)
        if init_as_mean:
            nn.init.zeros_(out_conv.weight)
            nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)
        self.f = nn.Sequential(*layers)
        # even kernels over-pad by one on the right; trim so T is preserved
        self._trim = 1 if self.kernel_size % 2 == 0 else 0
        if not init_as_mean:
            self._calibrate_branch(out_conv)

    @torch.no_grad()
    def _calibrate_branch(self, out_conv: nn.Conv1d, n_frames: int = 16) -> None:
        """Rescale the branch so a random init actually mixes frames.

        Default init leaves ``f(x)`` orders of magnitude below ``x`` when ``x`` is a
        unit-norm embedding (elements ~ ``1/sqrt(D)``), so an "un-initialised-as-mean"
        head would still be the mean to five decimals -- and a diagnostic run on it would
        report "no temporal effect" for a reason that has nothing to do with the data.
        One dummy forward fixes the scale empirically.
        """
        probe = torch.randn(4, self.in_dim, n_frames)
        probe = probe / probe.norm(dim=1, keepdim=True)
        h = self.f(probe)
        s = float(h.std())
        if s > 0:
            out_conv.weight.mul_(float(probe.std()) / s)
            out_conv.bias.mul_(float(probe.std()) / s)

    @property
    def order_sensitive(self) -> bool:  # noqa: D102
        # without per-position weights the trailing mean undoes the convolution
        return bool(self.n_frames)

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa: D102
        h = self.f(v.transpose(1, 2))  # (B, D, T)
        if self._trim:
            h = h[..., : v.shape[1]]
        return self.weights(v + h.transpose(1, 2))

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_dim={self.in_dim}, kernel_size={self.kernel_size}, n_frames={self.n_frames}, "
            f"pool={'position-weighted' if self.n_frames else 'mean (ORDER-BLIND: pass n_frames)'}"
        )


#: Frame poolers available to :class:`VideoSequenceProjector`.  ``mean`` is the
#: order-blind reference; the rest are order-sensitive (given ``learn_pos=True`` for
#: ``attn``) and all of them start out numerically equal to ``mean``.
FRAME_POOLINGS: Tuple[str, ...] = ("mean", "posw", "attn", "tconv")


class VideoSequenceProjector(nn.Module):
    """Project a *frozen video frame sequence* into the joint space.

    Contract F-seq: ``(B, T, D_vid) -> (B, D)``, L2-normalized -- the same output
    contract as :class:`VideoProjector`, so every loss, retrieval and evaluation path
    is unchanged.

    Input is ``cond_seq`` from ``data/derived/video_emb_seq/{tag}.npz`` (see
    :mod:`tactus.models.video.temporal`).  A ``(B, D_vid)`` input is accepted and treated
    as ``T = 1`` by the poolers that have no fixed frame count (``mean``); a pooler built
    for a specific ``T`` rejects it with an explicit error rather than silently pooling
    one frame.

    Parameters
    ----------
    in_dim:
        ``D_vid`` of the frozen encoder.
    out_dim:
        Joint-space dimensionality ``D``.
    n_frames:
        ``T`` of the cache.  Required for ``pool="posw"`` and for ``pool="attn"`` with
        learned positions; ignored by ``mean``/``tconv``.
    pool:
        One of :data:`FRAME_POOLINGS`.
    normalize_pooled:
        L2-normalize the pooled frame vector before the projector (default ``True``).
        With ``pool="mean"`` and unit-norm frames this makes the projector input exactly
        the contract-C embedding; turning it off changes the input scale and breaks
        comparability with the pooled arm.
    init_as_mean:
        Initialise the temporal pooler so it computes the plain mean (default ``True``).
    learn_pos:
        ``attn`` only.  ``False`` gives a content-based set-pooler, which is permutation
        invariant -- an order-blind control, not a temporal read-out.
    Remaining keywords (``hidden_dim``, ``n_layers``, ``dropout``, ``norm``, ``bias``,
    ``residual_blocks``) are forwarded verbatim to the inner :class:`VideoProjector`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        n_frames: Optional[int] = None,
        pool: str = "mean",
        hidden_dim: Optional[int] = None,
        n_layers: int = 1,
        dropout: float = 0.0,
        norm: Optional[str] = None,
        bias: bool = True,
        normalize: bool = True,
        residual_blocks: int = 0,
        normalize_pooled: bool = True,
        init_as_mean: bool = True,
        learn_pos: bool = True,
        attn_dim: Optional[int] = None,
        value_proj: bool = False,
        conv_channels: Optional[int] = None,
        kernel_size: int = 3,
        n_conv_layers: int = 1,
        pool_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pool = str(pool).lower()
        if pool not in FRAME_POOLINGS:
            raise ValueError(f"pool must be one of {FRAME_POOLINGS}, got {pool!r}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.pool = pool
        self.n_frames = int(n_frames) if n_frames else None
        self.normalize_pooled = bool(normalize_pooled)

        if pool == "mean":
            self.pooler: nn.Module = _MeanFramePool()
        elif pool == "posw":
            if not self.n_frames:
                raise ValueError("pool='posw' needs n_frames (the T of the sequence cache)")
            self.pooler = _PositionWeightFramePool(self.n_frames, init_as_mean=init_as_mean)
        elif pool == "attn":
            self.pooler = _AttentionFramePool(
                self.in_dim,
                n_frames=self.n_frames,
                attn_dim=attn_dim,
                learn_pos=bool(learn_pos),
                value_proj=bool(value_proj),
                init_as_mean=init_as_mean,
                dropout=float(pool_dropout),
            )
        else:  # tconv
            self.pooler = _TemporalConvFramePool(
                self.in_dim,
                n_frames=self.n_frames,
                conv_channels=conv_channels,
                kernel_size=kernel_size,
                n_layers=n_conv_layers,
                dropout=float(pool_dropout),
                init_as_mean=init_as_mean,
            )

        # the *same* head the pooled path uses, so pool="mean" is that path exactly
        self.proj = VideoProjector(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            norm=norm,
            bias=bias,
            normalize=normalize,
            residual_blocks=residual_blocks,
        )

    @property
    def order_sensitive(self) -> bool:
        """Can this head's output change when the frames are permuted?"""
        return bool(getattr(self.pooler, "order_sensitive", False))

    def pool_frames(self, v: torch.Tensor) -> torch.Tensor:
        """``(B, T, D_vid) -> (B, D_vid)``, the projector's input.  Exposed for probes."""
        if v.dim() == 2:
            v = v.unsqueeze(1)
        if v.dim() != 3:
            raise ValueError(f"VideoSequenceProjector expected (B, T, D_vid), got {tuple(v.shape)}")
        if v.shape[-1] != self.in_dim:
            raise ValueError(
                f"VideoSequenceProjector expected last dim {self.in_dim}, got {tuple(v.shape)}"
            )
        if self.n_frames and self.pool in ("posw", "tconv") and v.shape[1] != self.n_frames:
            raise ValueError(
                f"VideoSequenceProjector was built for T={self.n_frames}, got T={v.shape[1]}"
            )
        p = self.pooler(v)
        return l2_normalize(p) if self.normalize_pooled else p

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """``v``: ``(B, T, D_vid)`` (or ``(B, D_vid)``) -> ``(B, D)`` L2-normalized."""
        return self.proj(self.pool_frames(v))

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, pool={self.pool!r}, "
            f"n_frames={self.n_frames}, order_sensitive={self.order_sensitive}"
        )


#: Extra spellings for the sequence head on top of :data:`VIDEO_PROJECTOR_ALIASES`.
VIDEO_SEQ_PROJECTOR_ALIASES: Dict[str, Sequence[str]] = {
    **VIDEO_PROJECTOR_ALIASES,
    "n_frames": ("t", "n_steps", "seq_len", "num_frames"),
    "pool": ("pooling", "frame_pool", "readout"),
}

#: Named sequence-head shapes.  Each pairs a frame pooler with a projector shape.
VIDEO_SEQ_PROJECTOR_KINDS: Dict[str, Dict[str, Any]] = {
    "mean": {"pool": "mean", "n_layers": 1},
    "mean_mlp": {"pool": "mean", "n_layers": 2},
    "posw": {"pool": "posw", "n_layers": 1},
    "attn": {"pool": "attn", "n_layers": 1},
    "attn_mlp": {"pool": "attn", "n_layers": 2},
    "tconv": {"pool": "tconv", "n_layers": 1},
    "set": {"pool": "attn", "learn_pos": False, "n_layers": 1},  # order-blind control
}


def build_video_sequence_projector(cfg: Any = None, **overrides: Any) -> VideoSequenceProjector:
    """Build the frame-sequence projection head (contract F-seq).

    Same calling convention as :func:`build_video_projector` -- a run config with a
    ``video_projector`` / ``projector`` section, that section, or a bare kind name from
    :data:`VIDEO_SEQ_PROJECTOR_KINDS` -- plus keyword overrides.  Read ``in_dim`` and
    ``n_frames`` from the sequence cache rather than hard-coding them::

        z = np.load(f"{tag}.npz"); meta = json.loads(str(z["meta"]))
        head = build_video_sequence_projector(
            "attn", in_dim=meta["embedding_dim"], n_frames=meta["n_steps"], out_dim=256)
    """
    kind: Optional[str] = None
    if isinstance(cfg, str):
        kind, conf = cfg, {}
    else:
        conf = dict(
            unwrap_section(
                to_dict(cfg), "video_seq_projector", "video_projector", "video_head", "projector"
            )
        )
    conf.update(overrides)
    apply_aliases(conf, VIDEO_SEQ_PROJECTOR_ALIASES)
    conf.pop("_target_", None)
    params = conf.pop("params", None)
    if params is not None:
        merged = to_dict(params)
        merged.update(conf)
        conf = apply_aliases(merged, VIDEO_SEQ_PROJECTOR_ALIASES)
    kind = str(conf.pop("name", None) or conf.pop("kind", None) or kind or "mean").lower()

    for key, value in VIDEO_SEQ_PROJECTOR_KINDS.get(kind, {}).items():
        conf.setdefault(key, value)
    conf.setdefault("pool", kind if kind in FRAME_POOLINGS else "mean")
    if "in_dim" not in conf:
        raise KeyError(
            "build_video_sequence_projector needs in_dim (= D_vid of the frozen encoder). "
            "Read it from the sequence cache: "
            "json.loads(str(np.load(path)['meta']))['embedding_dim']"
        )
    kwargs = filter_kwargs(
        VideoSequenceProjector, conf, f"build_video_sequence_projector({kind!r})"
    )
    return VideoSequenceProjector(**kwargs)


# --------------------------------------------------------------------------------------
# time-resolved alignment
# --------------------------------------------------------------------------------------


def _ms_to_sample(t_ms: float, tmin_ms: float, sfreq: float) -> int:
    """Convert a millisecond offset into an index into the stored epoch."""
    return int(round((float(t_ms) - float(tmin_ms)) * float(sfreq) / 1000.0))


class TimeWindowHeads(nn.Module):
    """Run separate projection heads on EEG sub-windows and return a dict of embeddings.

    The trunk (temporal-spatial conv stack) is *shared* by default -- the
    convolutions are time-translation-equivariant so they apply to any window
    length -- while each window gets its own flatten-head projector, because the
    flattened feature dimension differs per window length.

    Returned dict keys are the window names plus (optionally) ``"full"``, which is
    the ordinary whole-epoch embedding produced by ``encoder.forward``.  Every
    value is ``(B, D)`` and L2-normalized, so each key can be fed straight into the
    loss contract.

    Parameters
    ----------
    encoder:
        A :class:`~tactus.models.eeg.base.EEGEncoder`.  Only the duck-typed API
        ``extract_features``/``feature_dim_for``/``conditioner``/``embed_dim`` is used,
        so any compatible trunk works.
    windows_ms:
        Mapping ``name -> (t_start_ms, t_stop_ms)``, half-open, relative to onset.
    window:
        Key into :data:`EPOCH_WINDOW_SPECS` giving ``tmin_ms``/``sfreq``.  Explicit
        ``tmin_ms``/``sfreq`` arguments override it.
    share_trunk:
        ``True`` (default) shares one trunk across windows; ``False`` deep-copies the
        trunk per window (more capacity, ~3x parameters, and the windows stop being
        comparable through a common representation -- prefer ``True``).
    include_full:
        Also return the whole-epoch embedding under key ``"full"``.
    head_factory:
        ``(in_dim, out_dim) -> nn.Module`` building each per-window head.  Defaults to
        a :class:`ResidualProjection` matching the encoder's own projector style.

    Notes
    -----
    Windows are *not* forced to be disjoint or contiguous; overlapping windows are
    legal and are sometimes wanted for a smoother onset curve.  A sliding-window
    family can be built with :meth:`sliding_windows`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        windows_ms: Mapping[str, Tuple[float, float]] = DEFAULT_TIME_WINDOWS_MS,
        window: str = "w0600",
        tmin_ms: Optional[float] = None,
        sfreq: Optional[float] = None,
        embed_dim: Optional[int] = None,
        share_trunk: bool = True,
        include_full: bool = True,
        dropout: float = 0.5,
        n_blocks: int = 1,
        head_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "extract_features") or not hasattr(encoder, "feature_dim_for"):
            raise TypeError(
                "TimeWindowHeads needs an EEGEncoder-like trunk exposing "
                "extract_features() and feature_dim_for(); got "
                f"{type(encoder).__name__}"
            )
        spec = EPOCH_WINDOW_SPECS.get(window, {})
        self.tmin_ms = float(tmin_ms if tmin_ms is not None else spec.get("tmin_ms", 0.0))
        self.sfreq = float(sfreq if sfreq is not None else spec.get("sfreq", 200.0))
        self.window = str(window)
        self.share_trunk = bool(share_trunk)
        self.include_full = bool(include_full)
        self.windows_ms = {str(k): (float(v[0]), float(v[1])) for k, v in windows_ms.items()}
        if not self.windows_ms:
            raise ValueError("windows_ms is empty")
        if "full" in self.windows_ms and include_full:
            raise ValueError("window name 'full' is reserved when include_full=True")

        self.encoder = encoder
        self.embed_dim = int(embed_dim if embed_dim is not None else getattr(encoder, "embed_dim", 256))
        n_times = int(getattr(encoder, "n_times", spec.get("n_times", 120)))
        self.n_times = n_times

        # ---- resolve millisecond windows into sample slices -----------------------
        slices: Dict[str, Tuple[int, int]] = {}
        for name, (t0, t1) in self.windows_ms.items():
            i0 = _ms_to_sample(t0, self.tmin_ms, self.sfreq)
            i1 = _ms_to_sample(t1, self.tmin_ms, self.sfreq)
            if i1 <= i0:
                raise ValueError(f"window {name!r} = ({t0}, {t1}) ms is empty at sfreq={self.sfreq}")
            if i0 < 0 or i1 > n_times:
                raise ValueError(
                    f"window {name!r} = ({t0}, {t1}) ms -> samples [{i0}, {i1}) falls outside the "
                    f"stored epoch [0, {n_times}) (window={self.window!r}, tmin={self.tmin_ms} ms)"
                )
            slices[name] = (i0, i1)
        self.window_slices = slices

        # ---- per-window trunks ----------------------------------------------------
        if share_trunk:
            self._extra_trunks = None
            trunks = {name: encoder for name in slices}
        else:
            copies = {}
            for name in slices:
                dup = copy.deepcopy(encoder)
                # deepcopy leaves the copy's SuLoRA adapters pointing at the original
                # conditioner; repair the links or they contribute nothing.
                if hasattr(dup, "relink_subject_conditioning"):
                    dup.relink_subject_conditioning()
                copies[name] = dup
            self._extra_trunks = nn.ModuleDict(copies)
            trunks = {name: self._extra_trunks[name] for name in slices}
        # plain dict on purpose: the modules are already registered via
        # ``self.encoder`` / ``self._extra_trunks`` and must not be registered twice.
        object.__setattr__(self, "_trunks", trunks)

        # ---- per-window heads -----------------------------------------------------
        if head_factory is None:

            def head_factory(in_dim: int, out_dim: int) -> nn.Module:  # type: ignore[misc]
                return ResidualProjection(
                    in_dim, out_dim, hidden_dim=out_dim, n_blocks=n_blocks, dropout=dropout, normalize=True
                )

        heads: Dict[str, nn.Module] = {}
        self.head_in_dims: Dict[str, int] = {}
        for name, (i0, i1) in slices.items():
            trunk = trunks[name]
            feat = int(trunk.feature_dim_for(i1 - i0))
            cond = getattr(trunk, "conditioner", None)
            if cond is not None:
                feat = int(cond.transform_feature_dim(feat))
            self.head_in_dims[name] = feat
            heads[name] = head_factory(feat, self.embed_dim)
        self.heads = nn.ModuleDict(heads)

        # Weight-level conditioning (SuLoRA) was attached to the encoder before these
        # heads existed, and the windowed path never runs the encoder's own projector.
        # Without this the SuLoRA arm would produce *unconditioned* window embeddings.
        self.adapted_head_modules: Dict[str, int] = {}
        for name in slices:
            cond = getattr(trunks[name], "conditioner", None)
            if cond is not None and hasattr(cond, "adapt_module"):
                self.adapted_head_modules[name] = int(cond.adapt_module(self.heads[name], f"heads.{name}"))

    # -- construction helpers ---------------------------------------------------------

    @staticmethod
    def sliding_windows(
        width_ms: float = 100.0,
        step_ms: float = 50.0,
        t_start_ms: float = 0.0,
        t_stop_ms: float = 600.0,
    ) -> Dict[str, Tuple[float, float]]:
        """Build a dense sliding-window family, e.g. for a fine alignment-onset curve.

        Returns ``{"w000_100": (0, 100), "w050_150": (50, 150), ...}``.
        """
        out: Dict[str, Tuple[float, float]] = {}
        t = float(t_start_ms)
        while t + width_ms <= t_stop_ms + 1e-9:
            out[f"w{int(round(t)):03d}_{int(round(t + width_ms)):03d}"] = (t, t + width_ms)
            t += float(step_ms)
        if not out:
            raise ValueError("sliding_windows produced no windows; check width/step/range")
        return out

    # -- forward ----------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, subject_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """``x``: ``(B, 64, T)`` -> ``{window_name: (B, D)}``, each L2-normalized."""
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        if x.shape[-1] != self.n_times:
            raise ValueError(
                f"TimeWindowHeads was built for T={self.n_times} (window={self.window!r}) "
                f"but received T={x.shape[-1]}"
            )
        out: Dict[str, torch.Tensor] = {}
        for name, (i0, i1) in self.window_slices.items():
            trunk = self._trunks[name]
            cond = getattr(trunk, "conditioner", None)
            xw = x[..., i0:i1]
            if cond is None:
                z = self.heads[name](trunk.extract_features(xw))
            else:
                # The per-window head is itself a SuLoRA target (see __init__), and
                # SuLoRALinear reads the active subject off the conditioner context.
                # Running the head outside that context leaves those adapters inert.
                with cond.subject_context(subject_id):
                    xw = cond.apply_input(xw, subject_id)
                    h = trunk.extract_features(xw)
                    h = cond.apply_features(h, subject_id)
                    z = self.heads[name](h)
            out[name] = l2_normalize(z)
        if self.include_full:
            out["full"] = self.encoder(x, subject_id)
        return out

    def extra_repr(self) -> str:  # noqa: D102
        wins = ", ".join(f"{k}={v[0]:.0f}-{v[1]:.0f}ms{self.window_slices[k]}" for k, v in self.windows_ms.items())
        return f"share_trunk={self.share_trunk}, include_full={self.include_full}, windows=[{wins}]"
