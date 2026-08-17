"""Weighted sum of registered sub-losses.

The blueprint's objective is a main cross-modal term plus small auxiliary terms
(``lambda_1`` CLISA, ``lambda_2`` RnC, ``lambda_3`` attribute anchors).  This
class is how that is assembled, from config, without any of the sub-losses
knowing they are being combined::

    loss:
      name: composite
      components:
        protonce: 1.0
        clisa:    0.3
        rnc:      0.1

Longhand gives per-component keyword arguments, an explicit ``type`` (so the
same loss can appear twice under different names), and per-component weight
schedules::

    loss:
      name: composite
      components:
        proto_cond:
          type: protonce
          weight: 1.0
          dim: 256
          granularities: [condition]
        proto_video:
          type: protonce
          weight: 0.5
          dim: 256
          granularities: [video]
        clisa:
          weight: 0.3
          warmup_steps: 2000     # ramp in after the main term has a foothold
          sequence_scope: subject

Logs from sub-loss ``k`` are prefixed ``k/``; the composite adds ``loss``,
``k/weight`` (the *effective* weight after any schedule) and ``k/raw_loss`` (the
sub-loss before weighting).  Reporting the raw value matters: a term whose
weight is 0.1 and whose raw loss is 40 is not an auxiliary term, it is the
objective, and only the unweighted number reveals that.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from .base import ContrastiveLoss, build_loss, register_loss, to_float


@register_loss("composite")
class CompositeLoss(ContrastiveLoss):
    """Sum of weighted sub-losses.

    Parameters
    ----------
    components
        Mapping from component name to either a scalar weight (the name is then
        also the loss type) or a dict of constructor kwargs, which may contain:

        ``type``          registered loss name (defaults to the component name)
        ``weight``        scalar multiplier (default 1.0)
        ``warmup_steps``  linear ramp from 0 to ``weight`` over this many steps
        ``start_step``    the component contributes nothing before this step

        Every remaining key is forwarded to the sub-loss constructor.
    normalize_weights
        Divide the total by the sum of the effective weights, keeping the loss
        magnitude comparable across configs with different numbers of terms.
    detach_zero_weight
        Skip the forward pass of components whose effective weight is exactly 0.
        Saves compute during a ``start_step`` delay, at the cost of leaving those
        components' parameters without gradient on those steps -- which trips
        DDP's unused-parameter detection, so it defaults to ``False``.
    """

    def __init__(
        self,
        components: Mapping[str, Union[float, int, Mapping[str, Any]]],
        normalize_weights: bool = False,
        detach_zero_weight: bool = False,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("composite loss needs at least one component")

        self.losses = nn.ModuleDict()
        self.weights: Dict[str, float] = {}
        self.warmup_steps: Dict[str, int] = {}
        self.start_steps: Dict[str, int] = {}
        self.normalize_weights = bool(normalize_weights)
        self.detach_zero_weight = bool(detach_zero_weight)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

        for name, spec in components.items():
            key = str(name)
            if "." in key:
                raise ValueError(
                    f"component name {key!r} must not contain '.' "
                    f"(nn.ModuleDict forbids it)"
                )
            if isinstance(spec, (float, int)):
                weight, sub_cfg = float(spec), {"name": key}
                warmup, start = 0, 0
            elif isinstance(spec, Mapping):
                cfg = dict(spec)
                weight = float(cfg.pop("weight", 1.0))
                warmup = int(cfg.pop("warmup_steps", 0))
                start = int(cfg.pop("start_step", 0))
                loss_type = cfg.pop("type", None) or cfg.pop("name", None) or key
                cfg["name"] = loss_type
                sub_cfg = cfg
            else:
                raise TypeError(
                    f"component {key!r} must be a number or a mapping, "
                    f"got {type(spec)!r}"
                )
            self.losses[key] = build_loss(sub_cfg)
            self.weights[key] = weight
            self.warmup_steps[key] = max(warmup, 0)
            self.start_steps[key] = max(start, 0)

        self.requires_video = any(
            getattr(m, "requires_video", True) for m in self.losses.values()
        )
        meta: list[str] = []
        for m in self.losses.values():
            meta.extend(getattr(m, "requires_meta", ()))
        self.requires_meta = tuple(dict.fromkeys(meta))

    # ------------------------------------------------------------------ #

    def step(self, n: int = 1) -> None:
        """Advance the internal step counter (call once per optimizer step).

        Only needed if any component uses ``warmup_steps`` or ``start_step``;
        harmless otherwise.
        """
        self._step += int(n)

    @property
    def current_step(self) -> int:
        return int(self._step.item())

    def effective_weight(self, name: str) -> float:
        """Weight of component ``name`` at the current step."""
        w = self.weights[name]
        step = self.current_step
        start = self.start_steps[name]
        if step < start:
            return 0.0
        warm = self.warmup_steps[name]
        if warm > 0:
            frac = min(1.0, max(0.0, (step - start) / float(warm)))
            w = w * frac
        return w

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        total: Optional[torch.Tensor] = None
        logs: Dict[str, float] = {}
        weight_sum = 0.0

        for name, module in self.losses.items():
            w = self.effective_weight(name)
            logs[f"{name}/weight"] = w
            if w == 0.0 and self.detach_zero_weight:
                logs[f"{name}/raw_loss"] = 0.0
                logs[f"{name}/skipped"] = 1.0
                continue

            out = module(z_eeg, z_vid, meta)
            if not isinstance(out, Mapping) or "loss" not in out:
                raise TypeError(
                    f"component '{name}' ({type(module).__name__}) returned "
                    f"{type(out)!r}; expected a dict with a 'loss' key. "
                    f"See the contract in tactus/losses/base.py."
                )
            sub_loss = out["loss"]
            for k, v in out.get("logs", {}).items():
                logs[f"{name}/{k}"] = v
            logs[f"{name}/raw_loss"] = to_float(sub_loss)

            contrib = w * sub_loss
            total = contrib if total is None else total + contrib
            weight_sum += abs(w)

        if total is None:
            total = self._zero_loss(z_eeg, z_vid)
        elif self.normalize_weights and weight_sum > 0.0:
            total = total / weight_sum

        logs["loss"] = to_float(total)
        logs["step"] = float(self.current_step)
        return {"loss": total, "logs": logs}

    def extra_repr(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.weights.items())
        return f"weights({parts}), normalize={self.normalize_weights}"


if __name__ == "__main__":  # pragma: no cover
    # importing the package registers every loss
    import tactus.losses  # noqa: F401
    from .base import make_dummy_batch

    torch.manual_seed(0)

    fn = CompositeLoss(
        components={
            "protonce": {"weight": 1.0, "dim": 256},
            "clisa": {"weight": 0.3, "warmup_steps": 4},
            "rnc": {"weight": 0.1, "label_key": "valence"},
        }
    )
    print(fn)
    for step in range(6):
        ze, zv, m = make_dummy_batch(batch_size=32, dim=256, seed=step)
        out = fn(ze, zv, m)
        out["loss"].backward()
        fn.step()
        print(
            f"[composite/step{step}] loss={out['loss'].item():.4f} "
            f"clisa_w={out['logs']['clisa/weight']:.3f} "
            f"proto_raw={out['logs']['protonce/raw_loss']:.3f} "
            f"clisa_raw={out['logs']['clisa/raw_loss']:.3f}"
        )

    # same loss twice under different names / granularities
    fn2 = CompositeLoss(
        components={
            "proto_cond": {"type": "protonce", "weight": 1.0, "dim": 256,
                           "granularities": ["condition"]},
            "proto_video": {"type": "protonce", "weight": 0.5, "dim": 256,
                            "granularities": ["video"]},
            "infonce": 0.2,
        }
    )
    ze, zv, m = make_dummy_batch(batch_size=32, dim=256)
    out = fn2(ze, zv, m)
    out["loss"].backward()
    print(f"[composite/two-protos] loss={out['loss'].item():.4f}")
    print("  log keys:", sorted(k for k in out["logs"] if k.endswith("raw_loss")))
