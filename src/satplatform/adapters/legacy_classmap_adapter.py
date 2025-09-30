## `src/satplatform/adapters/legacy_classmap_adapter.py`
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contracts.geo import GeoRaster
from ..contracts.core import ClassLabel
from ..ports.class_map import ClassMapPort, ClassMap

@dataclass(frozen=True)
class LegacyClassMapAdapter(ClassMapPort):
    """Construye ClassMap a partir de labels enteros (uint8/uint16).
    Requiere que `labels.data` sea 2D.
    """

    def from_labels(self, labels: GeoRaster, classes: Sequence[ClassLabel]) -> ClassMap:
        if labels.data.ndim != 2:
            raise ValueError("labels debe ser 2D")
        arr = labels.data
        ids = np.array([c.id for c in classes], dtype=arr.dtype)
        # Conteos
        vals, counts = np.unique(arr, return_counts=True)
        valid = {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist()) if int(v) in ids}
        # Paleta
        palette = {int(c.id): (c.color.r, c.color.g, c.color.b) for c in classes}
        return ClassMap(labels=labels, counts=valid, palette=palette)
