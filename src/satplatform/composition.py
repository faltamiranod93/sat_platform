### (Opcional) `composition/di.py` – Wiring rápido (si te sirve probar)
from __future__ import annotations

from ..adapters.gdal_raster_reader import GdalRasterReader
from ..adapters.gdal_raster_writer import GdalRasterWriter
from ..adapters.gdalwarp_cli import GdalWarpClipper
from ..adapters.csv_exporter import CSVExporter
from ..adapters.legacy_histnorm_adapter import LegacyHistNormAdapter
from ..adapters.legacy_pixelclass_adapter import LegacyPixelClassifier
from ..adapters.legacy_classmap_adapter import LegacyClassMapAdapter

from ..contracts.core import ClassLabel, MacroClass

# Factories simples
reader = GdalRasterReader()
writer = GdalRasterWriter()
clipper = GdalWarpClipper(raster_reader=reader, raster_writer=writer)
reporter = CSVExporter()
preproc = LegacyHistNormAdapter()
classes = [
    ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
    ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
    ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
]
classifier = LegacyPixelClassifier(classes_def=classes)
classmap = LegacyClassMapAdapter()