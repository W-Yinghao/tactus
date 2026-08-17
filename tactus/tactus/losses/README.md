# HOW TO ADD YOUR ALGORITHM

This package is the swap point. A new contrastive objective is **one file in this
directory plus one import line**. Nothing else in TACTUS needs to change — not
the trainer, not the eval harness, not the config schema.

```
tactus/losses/
  base.py              contract + registry + shared numerics   <- read this once
  infonce.py           CLIP-style symmetric InfoNCE (reference implementation)
  masked_infonce.py    + in-batch false-negative masking
  supcon.py            SupCon, multi-positive by label key
  protonce.py          EMA prototype bank, condition + video    <- PRIMARY objective
  softclip.py          KL against an external similarity matrix
  siglip.py            pairwise sigmoid, per-pair labels
  rnc.py               Rank-N-Contrast for continuous labels
  clisa.py             cross-subject, EEG-only
  composite.py         weighted sum of any of the above
```

---

## 1. The 15-line template

Copy this into `tactus/losses/my_loss.py` and edit the middle:

```python
import torch
import torch.nn.functional as F
from .base import ContrastiveLoss, TemperatureMixin, register_loss, to_float

@register_loss("my_loss")                       # <- the name used in configs
class MyLoss(ContrastiveLoss, TemperatureMixin):
    requires_video = True                       # False for EEG-only objectives
    requires_meta = ("condition_id",)           # checked before training starts

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = True):
        super().__init__()
        self._init_temperature(temperature, learnable=learnable_temperature)

    def forward(self, z_eeg, z_vid, meta):
        scale = self.scale()                    # clamped, differentiable
        logits = scale * (z_eeg @ z_vid.t())    # (B, B)
        target = torch.arange(z_eeg.shape[0], device=z_eeg.device)
        loss = F.cross_entropy(logits, target)
        return {"loss": loss, "logs": {"loss": to_float(loss), "scale": to_float(scale)}}
```

Then add one line to `__init__.py` (keep it alphabetical):

```python
from .my_loss import MyLoss
```

That import is what runs the decorator. A file that exists on disk but is not
imported there will fail with `unknown loss 'my_loss'`.

Finally, verify:

```bash
python -m tactus.losses --names my_loss     # adversarial battery, must all pass
python -m tactus.losses                     # everything, before you commit
```

---

## 2. The contract

**Signature.** `forward(z_eeg, z_vid, meta) -> dict`

| arg | shape | notes |
|---|---|---|
| `z_eeg` | `(B, D)` float | L2-normalized EEG embedding |
| `z_vid` | `(B, D)` float | L2-normalized **projected frozen** video embedding of the trial's condition |
| `meta`  | `dict[str, Tensor]` | each `(B,)` |

**Return.** `{"loss": <0-dim tensor>, "logs": {str: float}}`. Values in `logs`
must be plain Python floats — a live tensor there keeps the autograd graph alive
across the whole epoch. Use `to_float()`.

**`meta` keys** (`tactus.losses.base.META_KEYS`):

| key | type | range / meaning |
|---|---|---|
| `video_id` | long | 1..90, base video (orientation-independent) |
| `condition_id` | long | 0..359, `= (video_id - 1) * 4 + orientation` |
| `orientation` | long | 0 original, 1 horflip, 2 vertflip, 3 horvertflip |
| `subject_id` | long | 1..80 |
| `sequence_id` | long | 1..32, **within subject** — see gotcha 2 |
| `material_id` | long | 0..7 |
| `touch_type_id` | long | 0..11 |
| `toucher_id` | long | 0 hand, 1 object |
| `pain` | long | 0/1 |
| `valence`, `arousal`, `threat` | float | z-scored, constant within a base video |

Fetch them with `get_meta(meta, "condition_id", device=..., batch_size=B)`, which
handles dtype/device coercion and raises a useful error on a missing key rather
than silently substituting zeros.

**Helpers in `base.py`** worth reusing instead of rewriting:
`safe_normalize`, `safe_pdist`, `pairwise_eq`, `masked_log_softmax`,
`masked_logsumexp`, `mask_value`, `self._zero_loss(...)`, `self._prepare(...)`,
`make_dummy_batch(...)`.

---

## 3. Referencing it from config

Short form, when the loss needs no arguments:

```yaml
loss:
  name: my_loss
```

With arguments — every key other than `name` is forwarded to `__init__`, so a
typo raises `TypeError: unexpected keyword argument` instead of being ignored:

```yaml
loss:
  name: my_loss
  temperature: 0.05
  learnable_temperature: false
```

Combined with other terms:

```yaml
loss:
  name: composite
  components:
    protonce: 1.0          # shorthand: component name == loss name, value == weight
    my_loss:
      weight: 0.3
      temperature: 0.05
      warmup_steps: 2000   # linear ramp, needs composite.step() once per optimizer step
```

The same loss can appear twice under different names by giving an explicit
`type`:

```yaml
components:
  proto_cond:  {type: protonce, weight: 1.0, dim: 256, granularities: [condition]}
  proto_video: {type: protonce, weight: 0.5, dim: 256, granularities: [video]}
```

In code: `build_loss(cfg["loss"])`. Composite log keys are prefixed with the
component name (`proto_cond/condition_acc`), plus `<name>/weight` and
`<name>/raw_loss`.

---

## 4. The three gotchas

### Gotcha 1 — false negatives are the normal case, not an edge case

80 subjects x 360 conditions x 8 repeats means **any useful batch contains
duplicate conditions**, and because the video tower is frozen, two trials of the
same condition have *byte-identical* `z_vid`. Plain InfoNCE then asks the encoder
to rank `z_vid[i]` above `z_vid[j]` when the two vectors are equal — unsatisfiable,
so it contributes pure gradient noise and an irreducible loss floor.

Your options, in rough order of preference:

- put duplicates in the **positive** set (`supcon`, `siglip` with `positive_key`),
- **mask** them out of the denominator (`masked_infonce`),
- sidestep the issue by contrasting against a **prototype bank** rather than the
  batch (`protonce` — the primary objective, and immune by construction),
- give them **soft** targets (`softclip`, where equal conditions get equal mass
  automatically).

What you must not do is ignore it. `infonce` logs `false_neg_rate` precisely so
the cost of ignoring it stays visible.

There is a second, deeper level: the four orientations of one base video are
different stimuli but the same content. Whether they are positives, negatives, or
neither is a **scientific fork, not a hyperparameter** — treating flips as
negatives pushes capacity toward low-level/eye-movement axes, treating them as
positives discards the equivariance signal the orientation analysis depends on.
Every relevant loss exposes the switch (`mask_same_video`, `positive_key:
video_id`, `granularities`, `ignore_same_video`, `mask_sibling_orientations`).
Decide it deliberately and record the decision.

### Gotcha 2 — batch composition silently defines your loss

An in-batch objective's difficulty is set by whatever the sampler drew. Three
consequences specific to this dataset:

- **Slow drift.** Two trials from the same recording sequence share
  low-frequency artifacts (impedance, sweat, alertness). A contrastive loss will
  happily separate them using drift and learn nothing transferable. `clisa`
  therefore **refuses** negatives sharing the anchor's `(subject_id,
  sequence_id)`. Note the conjunction: `sequence_id` is 1..32 *within* a subject,
  so subject 3 sequence 5 and subject 40 sequence 5 are unrelated recordings, and
  refusing on `sequence_id` alone would throw away good negatives. That is what
  `sequence_scope="subject"` means; `"global"` is only correct if the trial table
  has been rewritten with globally unique sequence ids.
- **Cross-subject positives must actually be sampled.** `clisa` needs the same
  condition from different subjects *in the same batch*. With a naive random
  sampler that essentially never happens and the term silently contributes zero.
  Watch `clisa/n_valid` — if it is near 0, the sampler is wrong, not the loss.
- **Loss scale drifts with the sampler.** Comparing two runs whose batch
  composition differs compares samplers, not objectives. Prototype-based terms
  are the robust choice here because their negative set is the full gallery every
  step regardless of what was drawn.

Also, adjacent trials share raw samples at an 800 ms SOA and post-target trials
carry buttonpress motor potentials. Those are handled upstream in the split and
trial-table logic — but if a loss ever starts consuming raw time indices, it
inherits the problem.

### Gotcha 3 — temperature

- The stored parameter is `logit_scale = log(1 / temperature)` (CLIP convention).
  Read the multiplier with `self.scale()`, which clamps to `[min_scale,
  max_scale]` *inside the forward pass*, differentiably. Do not add an external
  `clamp_` in the train loop; do not read `self.logit_scale` directly.
- CLIP's ceiling of 100 is the default. Without it, a learnable temperature
  sharpens without bound early in training and the run collapses quietly.
- Not every loss wants the CLIP range. RnC runs near `temperature = 2`, i.e. a
  scale of 0.5, which is *below* CLIP's floor of 1 — hence the configurable
  `min_scale`, and hence RnC defaulting to a non-learnable temperature.
- **In a composite, every sub-loss owns its own learnable temperature.** Two
  learnable temperatures applied to the same similarity matrix are partially
  redundant and can drift in opposite directions. Prefer fixing all but one, and
  always log `logit_scale` per component.
- Single-trial EEG has low SNR; a temperature tuned on pseudo-trial (k=4
  averaged) batches will be wrong for single-trial batches. Re-tune when the
  curriculum anneals.

---

## 5. Invariants your loss must satisfy

The battery in `python -m tactus.losses` enforces these, and it is the first
thing to run after any edit:

1. **Never NaN.** Not on a batch where every sample shares one condition, not on
   a single-subject batch, not on `B = 1`. Mask with `mask_value(dtype)` (a large
   finite negative), never `-inf` — a fully-masked row of `-inf` returns NaN from
   `log_softmax` and poisons the whole batch even if you discard the row.
2. **Drop invalid rows, do not average them in.** A row with no positive or no
   negative contributes nothing meaningful; dividing by `B` instead of by the
   number of valid rows makes the loss depend on batch composition.
3. **Return a graph-connected zero when no row is valid.** Use
   `self._zero_loss(z_eeg, z_vid)`. A bare `torch.tensor(0.0)` breaks
   `.backward()`, leaves `.grad` as `None`, and trips DDP's unused-parameter
   check.
4. **Log a validity counter.** Every loss here reports `n_valid` and
   `degenerate`. A term that is quietly contributing zero on 90% of steps looks
   exactly like a term that is working, unless you log it.
