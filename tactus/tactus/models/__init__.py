"""TACTUS model zoo: EEG encoders, video projection heads, subject conditioning, EA.

Public builders
---------------
``build_eeg_encoder(cfg)``
    Build an :class:`~tactus.models.eeg.base.EEGEncoder` from a config mapping.
``build_video_projector(cfg)``
    Build the frozen-video-embedding head (contract F).
``build_time_window_heads(encoder, cfg)``
    Wrap an encoder in the time-resolved alignment heads.

Config shape (plain dict, ``argparse.Namespace``, or OmegaConf -- all accepted)::

    model:
      name: tsconv                 # tsconv | atm  (see list_eeg_encoders())
      n_channels: 64
      n_times: 120                 # 120 for w0600 (primary), 180 for wm100_800
      embed_dim: 256
      n_subjects: 80
      subject_cond:                # or just: subject_cond: subject_layer
        name: subject_layer        # none | subject_token | subject_layer | sulora
        rank: 16
      n_filters: 40                # architecture-specific keys pass straight through
    video_projector:
      in_dim: 768                  # D_vid of the frozen encoder
      out_dim: 256                 # must equal model.embed_dim
      n_layers: 1

Keys the target class does not accept are dropped with a warning rather than raising,
so a shared config can carry keys for several architectures.
"""

from __future__ import annotations

from typing import Any, Dict

from ._cfg import filter_kwargs, to_dict, unwrap_section
from .ea import EAApplier, EAState, EuclideanAlignment, inverse_sqrt_psd
from .eeg.base import (
    EEGEncoder,
    build_eeg_encoder,
    count_parameters,
    get_eeg_encoder,
    list_eeg_encoders,
    register_eeg_encoder,
)
from .eeg.subject_cond import (
    UNSEEN_SUBJECT,
    SubjectConditioner,
    SubjectLayer,
    SubjectToken,
    SuLoRA,
    build_subject_conditioner,
    list_subject_conditioners,
    register_subject_conditioner,
)
from .heads import (
    DEFAULT_TIME_WINDOWS_MS,
    EPOCH_WINDOW_SPECS,
    ResidualProjection,
    TimeWindowHeads,
    VideoProjector,
    build_video_projector,
    l2_normalize,
)

# importing the architecture modules is what populates the encoder registry
from .eeg import atm as _atm  # noqa: F401
from .eeg import tsconv as _tsconv  # noqa: F401

__all__ = [
    "build_eeg_encoder",
    "build_video_projector",
    "build_time_window_heads",
    "EEGEncoder",
    "VideoProjector",
    "TimeWindowHeads",
    "ResidualProjection",
    "l2_normalize",
    "register_eeg_encoder",
    "get_eeg_encoder",
    "list_eeg_encoders",
    "count_parameters",
    "SubjectConditioner",
    "SubjectToken",
    "SubjectLayer",
    "SuLoRA",
    "build_subject_conditioner",
    "list_subject_conditioners",
    "register_subject_conditioner",
    "UNSEEN_SUBJECT",
    "EuclideanAlignment",
    "EAState",
    "EAApplier",
    "inverse_sqrt_psd",
    "DEFAULT_TIME_WINDOWS_MS",
    "EPOCH_WINDOW_SPECS",
]


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
#
# build_eeg_encoder lives in tactus.models.eeg.base and build_video_projector in
# tactus.models.heads -- the modules that own the classes, and the import paths the
# trainer probes first.  They are re-exported here for convenience.


def build_time_window_heads(encoder: EEGEncoder, cfg: Any = None, **overrides: Any) -> TimeWindowHeads:
    """Wrap ``encoder`` in per-window projection heads for the time-resolved analysis.

    ``cfg`` may carry ``windows_ms`` as a mapping, or ``sliding: {width_ms, step_ms}``
    to generate a dense family via :meth:`TimeWindowHeads.sliding_windows`.
    """
    conf = dict(unwrap_section(to_dict(cfg), "time_windows", "time_window_heads"))
    conf.update(overrides)
    conf.pop("_target_", None)
    sliding = conf.pop("sliding", None)
    if sliding is not None:
        conf["windows_ms"] = TimeWindowHeads.sliding_windows(**to_dict(sliding))
    elif "windows_ms" in conf:
        conf["windows_ms"] = {k: tuple(v) for k, v in to_dict(conf["windows_ms"]).items()}
    conf.setdefault("embed_dim", encoder.embed_dim)
    kwargs = filter_kwargs(TimeWindowHeads, conf, "build_time_window_heads")
    kwargs.pop("encoder", None)
    return TimeWindowHeads(encoder, **kwargs)
