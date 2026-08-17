"""EEG encoder tower: architectures, the registry, and subject conditioning.

Architectures
-------------
``tsconv`` (aliases ``nice``, ``shallow``)
    NICE-style shallow temporal-spatial convolution, ~0.33 M parameters.  The primary
    baseline.
``atm``
    Channel self-attention front end + the same conv trunk + a wide residual
    projector, ~3.2 M parameters.

Subject conditioning is orthogonal to the architecture: any encoder accepts
``subject_cond`` in ``{"none", "subject_token", "subject_layer", "sulora"}``, and each
mechanism carries a fixed a-priori rule for subjects it has never seen.  See
:mod:`tactus.models.eeg.subject_cond`.
"""

from __future__ import annotations

from .base import (
    EEGEncoder,
    build_eeg_encoder,
    count_parameters,
    get_eeg_encoder,
    list_eeg_encoders,
    register_eeg_encoder,
)
from .subject_cond import (
    UNSEEN_SUBJECT,
    NoSubjectConditioning,
    SubjectConditioner,
    SubjectLayer,
    SubjectToken,
    SuLoRA,
    build_subject_conditioner,
    list_subject_conditioners,
    register_subject_conditioner,
)
from .atm import ATMEncoder, ChannelAttention, TimeAttention
from .tsconv import TemporalSpatialConv, TSConvEncoder

# --------------------------------------------------------------------------- #
# Foundation-model wrappers (baseline-ladder rung 5).
#
# Registration happens as an import side effect, so these have to be imported
# here or `build_eeg_encoder("labram")` raises KeyError.  They are imported
# DEFENSIVELY: they need `braindecode` plus a pretrained checkpoint, both of
# which are optional heavy dependencies, and `tactus.models.eeg` is on the
# import path of the trainer, the selftest and every baseline.  A missing or
# broken optional dependency must degrade to "that encoder is unavailable",
# never to "the whole package fails to import".
# --------------------------------------------------------------------------- #
FM_ENCODERS_AVAILABLE: dict[str, str] = {}

for _mod, _cls in (("fm_labram", "LabramEncoder"),
                   ("fm_cbramod", "CBraModEncoder"),
                   ("fm_eegpt", "EEGPTEncoder")):
    try:
        _m = __import__(f"{__name__}.{_mod}", fromlist=[_cls])
        globals()[_cls] = getattr(_m, _cls)
        FM_ENCODERS_AVAILABLE[_mod] = "ok"
    except Exception as _exc:  # noqa: BLE001 - optional dependency
        FM_ENCODERS_AVAILABLE[_mod] = f"{type(_exc).__name__}: {_exc}"
del _mod, _cls

__all__ = [
    "EEGEncoder",
    "build_eeg_encoder",
    "register_eeg_encoder",
    "get_eeg_encoder",
    "list_eeg_encoders",
    "count_parameters",
    "TSConvEncoder",
    "TemporalSpatialConv",
    "ATMEncoder",
    "ChannelAttention",
    "TimeAttention",
    "SubjectConditioner",
    "NoSubjectConditioning",
    "SubjectToken",
    "SubjectLayer",
    "SuLoRA",
    "build_subject_conditioner",
    "list_subject_conditioners",
    "register_subject_conditioner",
    "UNSEEN_SUBJECT",
    "FM_ENCODERS_AVAILABLE",
]
