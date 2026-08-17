"""Video tower: the frozen-encoder embedding cache (contract C).

The video side of TACTUS is *frozen*.  Nothing here trains; :mod:`encode` runs a
pre-trained image/video encoder over the 360 stimulus conditions once and writes
``data/derived/video_emb/{model_tag}.npz``.  The only learned video-side parameters
live in :class:`tactus.models.heads.VideoProjector`.

Loading a cache::

    import json, numpy as np
    z = np.load("data/derived/video_emb/siglip2-base.npz")
    cond_emb = z["cond_emb"]                    # (360, D), L2-normalized
    base_emb = z["base_emb"]                    # (90, D)
    meta = json.loads(str(z["meta"].item()))    # model, pooling, n_frames, provenance

Building one::

    python -m tactus.models.video.encode --list-models
    python -m tactus.models.video.encode --stim-root ds005662 --model siglip2-base
"""

from __future__ import annotations

from .encode import (
    MODEL_REGISTRY,
    ORIENTATION_NAMES,
    ModelSpec,
    build_cache,
    condition_id,
    discover_orientation_dirs,
    parse_orientation,
    parse_video_id,
    read_video_frames,
)

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "ORIENTATION_NAMES",
    "condition_id",
    "parse_orientation",
    "parse_video_id",
    "discover_orientation_dirs",
    "read_video_frames",
    "build_cache",
]
