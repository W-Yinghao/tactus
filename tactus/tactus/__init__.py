"""TACTUS -- Touch Alignment by Contrastive Transfer to Unseen Subjects.

EEG-video contrastive learning on OpenNeuro ds005662 (80 subjects x 360
conditions x 8 repeats).

Subpackages
-----------
``tactus.data``       download / event parsing / preprocessing / splits / datasets
``tactus.models``     EEG encoders and projection heads
``tactus.losses``     the swappable contrastive-objective registry
``tactus.eval``       retrieval, RSA, noise ceilings, permutation tests
``tactus.baselines``  linear MVPA and alignment baselines
``tactus.train``      training loop and run driver

Nothing is imported eagerly here: the subpackages have very different dependency
footprints (``tactus.data.events`` needs only pandas, ``tactus.data.preprocess``
needs MNE, ``tactus.models`` needs torch), and a metadata-only run on a laptop
must not be blocked by a missing GPU stack.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
