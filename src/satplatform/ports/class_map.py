# src/satplatform/ports/class_map.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Protocol, runtime_checkable
from ..contracts.core import ClassLabel
from ..contracts.geo import GeoRaster

@dataclass(frozen=True)
class ClassMap:
    """Producto discreto de clases por píxel."""
    labels: GeoRaster                      # enteros (id de clase)
    counts: Mapping[int, int]              # id -> número de píxeles
    palette: Mapping[int, tuple[int,int,int]]  # id -> (R,G,B)

@runtime_checkable
class ClassMapPort(Protocol):
    """
    Construcción/posprocesamiento de classmap a partir de labels o probabilidades.
    """
    def from_labels(self, labels: GeoRaster, classes: Sequence[ClassLabel]) -> ClassMap: ...
    # Si más adelante manejas probabilidades, agrega:
    # def from_probas(self, probas: np.ndarray, classes: Sequence[ClassLabel]) -> ClassMap: ...

__all__ = ["ClassMapPort", "ClassMap"]
