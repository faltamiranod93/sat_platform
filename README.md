# sat-platform

> Plataforma modular para procesamiento de imágenes multiespectrales Sentinel-2 aplicada a tranques de relaves y monitoreo ambiental.

---

## Estado actual del proyecto

> **Actualizado: mayo 2026 (post-Sprint 2)**
> Este README refleja el estado real del código tras la corrección de los bugs
> bloqueantes y el refactor del composition root.

| Componente | Estado | Notas |
|---|---|---|
| `contracts/core.py` | ✅ Estable | `RGB8`, `ClassLabel`, `SceneId`, `RunMeta`, `CalibrationSpec` completos |
| `contracts/geo.py` | ✅ Estable | `CRSRef`, `GeoProfile`, `GeoRaster`, `validate_profile_compat` (con manejo correcto de NaN nodata) |
| `contracts/products.py` | ✅ Estable | `BandSet`, `S2Asset` con inmutabilidad garantizada |
| `ports/` | ✅ Estable | 100% cobertura; Protocols definidos y documentados |
| `adapters/gdal_raster_reader.py` | ✅ Funcional | Fallback rasterio → GDAL → tifffile |
| `adapters/gdal_raster_writer.py` | ✅ Funcional | Escritura GeoTIFF/COG |
| `adapters/legacy_histnorm` | ✅ Funcional | RGB→HSL corregido (escalar común, preserva cromaticidad) |
| `adapters/legacy_pixelclass` | ✅ Funcional | Usa `b in bands.names()` (bug `bands.has()` corregido) |
| `services/classmap_service.py` | ✅ Funcional | Pipeline `run()` operativo con cobertura ≥75% |
| `services/preprocessing_service.py` | ✅ Funcional | Helpers a nivel módulo; `__all__` limpio (SyntaxError corregido) |
| `services/histogram_norm_service.py` | ✅ Funcional | `BandSet` importado correctamente |
| `services/spectral_service.py` | 🟡 Parcial | Solo 3 índices disponibles (NDVI, NDWI, NDBI). BSI/MNDWI/SAVI/EVI en Fase 5 |
| `services/training_service.py` | 🟡 Aislado | `TrainingService.build_dataset/split` listo, sin integración al pipeline aún |
| `config.py` | ✅ Estable | Un único `model_config` con `frozen=True` (corregido en Sprint 1) |
| `composition/di.py` | ✅ Funcional | Wiring completo: `build_classmap_service`, `build_preprocessing_service`, `build_histogram_norm_service`, etc. |
| `cli.py` | ✅ Funcional | `argparse` + delega a `di.py`; entry point `satplatform` registrado |
| `tests/` | 🟢 38 tests, 62% cobertura | Unit + integration con autogeneración de fixtures TIFF |
| CI/CD | ⬜ Pendiente | GitHub Actions planificado para Fase 8 |

---

## Tabla de contenidos

1. [¿Qué es sat-platform?](#qué-es-sat-platform)
2. [Arquitectura hexagonal](#arquitectura-hexagonal)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Instalación](#instalación)
5. [Configuración](#configuración)
6. [Flujo de trabajo completo](#flujo-de-trabajo-completo)
7. [CLI — referencia de comandos](#cli--referencia-de-comandos)
8. [Índices espectrales disponibles](#índices-espectrales-disponibles)
9. [Clases de cobertura](#clases-de-cobertura)
10. [Extender la plataforma](#extender-la-plataforma)
11. [Tests](#tests)
12. [Decisiones de diseño](#decisiones-de-diseño)
13. [Limitaciones conocidas](#limitaciones-conocidas)
14. [Contribuir](#contribuir)
15. [Entorno reproducible](#entorno-reproducible)
16. [Roadmap por fases](#roadmap-por-fases)
17. [Referencias](#referencias)
18. [Licencia](#licencia)

---

## ¿Qué es sat-platform?

`sat-platform` procesa imágenes multiespectrales de Sentinel-2 para producir classmaps de cobertura superficial sobre áreas de interés, con foco en **tranques de relaves** y monitoreo ambiental. El pipeline es determinista y reproducible: dadas las mismas entradas y configuración, siempre produce los mismos productos.

**Capacidades actuales:**

- Lectura de bandas Sentinel-2 (JP2, GeoTIFF, COG) vía rasterio o GDAL
- Recorte a región de interés (ROI) con GeoJSON o WKT
- Normalización por histograma (percent-clip, z-score, min-max, equalización, matching)
- Cálculo de índices espectrales (NDVI, NDWI, NDBI)
- Conversión RGB→HSL para features espectrales
- Clasificación por píxel mediante reglas o modelo externo
- Exportación de classmaps como GeoTIFF y quicklook PNG
- Gestión de configuración por proyecto vía `settings.yaml`

---

## Arquitectura hexagonal

El diseño sigue el patrón **Ports & Adapters**: el dominio no depende de ninguna librería externa. Las implementaciones concretas (GDAL, rasterio, sklearn) viven en adapters que se conectan vía ports en `composition/di.py`.

```
┌─────────────────────────────────────────────────────────────┐
│                          CLI / UI                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ usa
┌──────────────────────────▼──────────────────────────────────┐
│                  composition/di.py                          │
│          (wiring: ports ←→ adapters ←→ services)            │
└───────┬──────────────────┬──────────────────────────────────┘
        │                  │
┌───────▼──────┐   ┌───────▼──────────────────────────────────┐
│  adapters/   │   │              services/                   │
│  (GDAL,      │   │  (lógica pura: classmap, histogram,       │
│  rasterio,   │   │   preprocessing, spectral, training)      │
│  legacy...)  │   └───────────────────────────────────────────┘
└───────┬──────┘                    │ usa
        │ implementa                │
┌───────▼──────────────────────────▼──────────────────────────┐
│                        ports/                               │
│  (Protocols: RasterReader, RasterWriter, ROIClipper,        │
│   PixelClassifier, ClassMap, Preprocessing, Exporters,      │
│   Catalog)                                                  │
└─────────────────────────────────────────────────────────────┘
                           │ tipos de dominio
┌──────────────────────────▼──────────────────────────────────┐
│                      contracts/                             │
│   core.py: ClassLabel, SceneId, RGB8, RunMeta               │
│   geo.py:  GeoRaster, GeoProfile, CRSRef, Bounds            │
│   products.py: BandSet, S2Asset                             │
└─────────────────────────────────────────────────────────────┘
```

**Regla fundamental:** las flechas de dependencia solo apuntan hacia adentro. `services/` nunca importa de `adapters/`. `contracts/` no importa de ningún otro módulo del proyecto.

---

## Estructura del proyecto

### Código fuente

```
src/satplatform/
├── contracts/          # Tipos de dominio inmutables (Pydantic + dataclasses)
│   ├── core.py         # ClassLabel, SceneId, RGB8, RunMeta, CalibrationSpec
│   ├── geo.py          # GeoRaster, GeoProfile, CRSRef, Bounds, helpers
│   └── products.py     # BandSet, S2Asset, Band
├── ports/              # Interfaces (Protocol) de entrada/salida
│   ├── raster_read.py  # RasterReaderPort
│   ├── raster_write.py # RasterWriterPort
│   ├── roi.py          # ROIClipperPort
│   ├── preprocessing.py# PreprocessingPort, NormalizeSpec
│   ├── pixel_class.py  # PixelClassifierPort
│   ├── class_map.py    # ClassMapPort, ClassMap
│   ├── exporters.py    # QuicklookExporterPort, ReportExporterPort
│   └── catalog.py      # CatalogPort, ROIItem, MosaicItem, CatalogItem
├── services/           # Lógica de dominio pura (sin I/O)
│   ├── classmap_service.py       # pipeline LOAD→ALIGN→PRE→INFER→EXPORT
│   ├── preprocessing_service.py  # normalize_single, normalize_many, rgb_to_hsl
│   ├── histogram_norm_service.py # equalize, match, percent_clip
│   ├── spectral_service.py       # NDVI, NDWI, NDBI, rgb_to_hsl
│   └── training_service.py       # build_dataset, split, class_weights
├── adapters/           # Implementaciones concretas
│   ├── gdal_raster_reader.py     # fallback rasterio → GDAL → tifffile
│   ├── gdal_raster_writer.py     # escritura GeoTIFF/COG
│   ├── gdalwarp_cli.py           # recorte ROI vía gdalwarp
│   ├── legacy_histnorm_adapter.py
│   ├── legacy_pixelclass_adapter.py
│   ├── legacy_classmap_adapter.py
│   ├── legacy_fil2roi_adapter.py
│   ├── csv_catalog.py
│   └── csv_exporter.py
├── composition/
│   └── di.py           # Composition root: build_classmap_service, build_*_service
├── config.py           # Settings (Pydantic), validación de placeholders
└── cli.py              # Entrypoint argparse; delega a di.build_*_service
```

### Composition root (`composition/di.py`)

Todo el wiring adapters↔ports↔services vive aquí. `cli.py` y futuros consumidores
solo deben llamar a estos builders, nunca construir adapters/services directamente.

| Builder | Devuelve |
|---|---|
| `build_settings(project_root)` | `Settings` con `class_labels.json` fusionado |
| `resolve_classes(settings)` | tuple de `ClassLabel` (settings o defaults) |
| `build_raster_reader()` | `GdalRasterReader` (rasterio→GDAL→tifffile) |
| `build_raster_writer()` | `GdalRasterWriter` |
| `build_clipper(settings)` | `GdalWarpClipper` |
| `build_preprocessing_adapter()` | `LegacyHistNormAdapter` |
| `build_pixel_classifier(settings)` | `LegacyPixelClassifier` con clases inyectadas |
| `build_class_mapper()` | `LegacyClassMapAdapter` |
| `build_classmap_service(settings)` | `ClassMapService` con todos los puertos cableados |
| `build_preprocessing_service(settings)` | `PreprocessingService` cableado |
| `build_histogram_norm_service(settings)` | `HistogramNormService` cableado |
| `build_spectral_service()` | `SpectralService` (puro dominio) |
| `build_training_service()` | `TrainingService` (puro dominio) |

### Estructura de proyectos (layout físico)

Cada proyecto de monitoreo sigue esta convención:

```
Proyecto/
├── 00-Config/
│   ├── settings.yaml       # paths, CRS, patrones, band_order
│   ├── class_labels.json   # catálogo de clases (id, nombre, color RGB, macro)
│   └── roi_master.geojson  # AOIs maestras del proyecto
├── 01-Raw/                 # Datos crudos — NO modificar
│   ├── s2/                 # Productos SAFE de Sentinel-2
│   │   └── S2A_MSIL2A_20240123T143731_N0510_R096_T19HFE_20240123T181234.SAFE/
│   ├── DEM/                # Modelos digitales de elevación
│   └── Ancillary/          # Capas auxiliares (límites, catastro)
├── 02-Work/                # Artefactos intermedios (recalculables)
│   ├── ROI/                # Bandas recortadas por ROI
│   ├── STACK/              # Stacks multibanda por fecha
│   ├── HIST-NORM/          # Bandas normalizadas
│   ├── FEATURES/           # Features espectrales (HSL, índices)
│   └── CLASSMAP-WORK/      # Classmaps intermedios
├── 03-Products/            # Productos finales
│   ├── CLASSMAP/           # GeoTIFF de clases por fecha
│   ├── CLASS-VIS/          # Visualizaciones coloreadas
│   ├── VIS/                # Quicklooks RGB
│   └── REPORT/             # Reportes CSV/HTML por fecha
└── 04-Analysis/            # Notebooks, scripts exploratorios, figuras
```

---

## Instalación

### Requisitos previos

| Requisito | Versión mínima | Obligatorio |
|---|---|---|
| Python | 3.11 | Sí |
| rasterio | 1.3 | Recomendado |
| GDAL | 3.4 | Alternativa a rasterio |
| gdalwarp | incluido en GDAL | Para recorte ROI |

### Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/faltamiranod93/sat_platform.git
cd sat_platform

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instalar en modo editable con dependencias de desarrollo
pip install -e ".[dev]"

# 4. Verificar instalación
satplatform --help        # entry point registrado en pyproject.toml
# o equivalente:
python -m satplatform.cli --help
```

### Extras de instalación disponibles

| Extra | Contenido | Cuándo usarlo |
|---|---|---|
| `[tests]` | `pytest`, `pytest-cov` | CI mínimo |
| `[dev]` | tests + `ruff`, `mypy`, `pre-commit` | Desarrollo |
| `[raster]` | `rasterio>=1.3` | Producción (lectura georreferenciada) |

### Sin GDAL (modo degradado para desarrollo)

Las dependencias base (`pyproject.toml`) ya incluyen `tifffile` y `Pillow`, así que la
suite de tests corre **sin GDAL/rasterio**. En producción, instala el extra `[raster]`:

```bash
pip install -e ".[dev,raster]"
```

### Verificación rápida

```bash
# Verifica que los contratos cargan sin error
python -c "from satplatform.contracts.geo import GeoRaster, GeoProfile; print('OK')"

# Verifica que los adapters detectan el backend disponible
python -c "from satplatform.adapters.gdal_raster_reader import GdalRasterReader; print('OK')"
```

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---|---|---|
| `pydantic` | >=2.0 | Contratos de dominio y validación |
| `pydantic-settings` | >=2.0 | Carga de `Settings` desde YAML y variables de entorno |
| `rasterio` | >=1.3 | I/O raster (preferido sobre GDAL directo) |
| `numpy` | >=1.24 | Operaciones numéricas sobre arrays |
| `Pillow` | >=10.0 | Exportación de quicklooks PNG |
| `PyYAML` | >=6.0 | Lectura de `settings.yaml` |
| `gdal` / `osgeo` | >=3.4 | Backend alternativo a rasterio |
| `tifffile` | >=2023.0 | Fallback mínimo para tests sin GDAL |

**Dependencias de desarrollo:**

| Paquete | Uso |
|---|---|
| `pytest` | Suite de tests |
| `pytest-cov` | Cobertura de tests |
| `ruff` | Linting y formato |
| `mypy` | Type checking estricto |
| `pre-commit` | Hooks pre-commit |

---

## Configuración

### `00-Config/settings.yaml`

```yaml
project_root: "."
crs_out: "EPSG:32719"

# Directorios de trabajo (relativos a project_root)
work_roi_dir: "02-Work/ROI"
work_products_dir: "03-Products"
report_dir: "03-Products/REPORT"

# Herramientas externas (null = busca en PATH)
gdalwarp_exe: null

# Orden de bandas por defecto para stacks
band_order: ["B02", "B03", "B04"]

# Las clases pueden definirse aquí o en class_labels.json
classes: []

# Patrones de entrada — placeholders: {product}, {granule}, {tile}, {sensing}, {band}, {res}
input_patterns:
  safe_dir:    "01-Raw/s2/{product}.SAFE"
  granule_dir: "01-Raw/s2/{product}.SAFE/GRANULE/{granule}"
  jp2_file:    "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_{band}_{res}.jp2"
  scl_file:    "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_SCL_{res}.jp2"
  mask_file:   "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/QI_DATA/MSK_QUALIT_{band}.jp2"
  roi_file:    "00-Config/roi_master.geojson"

# Patrones de salida — placeholder: {date} (YYYYMMDD)
output_patterns:
  stack:         "02-Work/STACK/{date}/stack.tif"
  hist_norm:     "02-Work/HIST-NORM/{date}/hn.tif"
  features_hsl:  "02-Work/FEATURES/{date}/hsl.tif"
  classmap:      "03-Products/CLASSMAP/{date}/classmap.tif"
```

### `00-Config/class_labels.json`

```json
[
  {"id": 1, "name": "Agua",    "macro": "Agua",    "color": {"r": 31,  "g": 119, "b": 180}},
  {"id": 2, "name": "Relave",  "macro": "Relave",  "color": {"r": 214, "g": 39,  "b": 40}},
  {"id": 3, "name": "Terreno", "macro": "Terreno", "color": {"r": 152, "g": 223, "b": 138}}
]
```

### Variables de entorno

El proyecto soporta configuración parcial vía variables de entorno con prefijo `SAT_`:

```bash
export SAT_PROJECT_ROOT=/ruta/al/proyecto
export SAT_CRS_OUT=EPSG:32719
export SAT_GDALWARP_EXE=/usr/bin/gdalwarp
```

---

## Flujo de trabajo completo

El pipeline transforma bandas crudas en un classmap georreferenciado siguiendo estos pasos:

```
Bandas S2 (JP2/TIF)
       │
       ▼
  [1] LOAD      — lee cada banda con GdalRasterReader
       │
       ▼
  [2] CLIP      — recorta a ROI con GdalWarpClipper (opcional)
       │
       ▼
  [3] ALIGN     — valida compatibilidad de grid, CRS y dimensiones
       │
       ▼
  [4] NORMALIZE — histograma percent-clip 2-98 (opcional, --normalize)
       │
       ▼
  [5] BANDSET   — agrupa bandas en BandSet con resolución verificada
       │
       ▼
  [6] CLASSIFY  — LegacyPixelClassifier (reglas) o modelo externo
       │
       ▼
  [7] CLASSMAP  — cuenta píxeles por clase, construye paleta RGB
       │
       ▼
  [8] EXPORT    — GeoTIFF uint8 + quicklook PNG (opcional)
```

### Ejemplo completo: bandas crudas → classmap

```bash
# Pipeline completo con recorte ROI, normalización y quicklook
python -m satplatform.cli classify \
  --date 20240123 \
  -b B03=./01-Raw/s2/T19HFE_B03_10m.tif \
  -b B04=./01-Raw/s2/T19HFE_B04_10m.tif \
  -b B08=./01-Raw/s2/T19HFE_B08_10m.tif \
  -b B11=./01-Raw/s2/T19HFE_B11_20m.tif \
  --roi   ./00-Config/roi_master.geojson \
  --normalize \
  --png \
  --out   ./03-Products/CLASSMAP/20240123/classmap.tif
```

Output esperado:

```
[INFO] Leyendo 4 bandas...
[INFO] Recortando a ROI...
[INFO] Normalizando (percent-clip 2-98)...
[INFO] Clasificando 1.152.000 píxeles...
[OK]   classmap.tif → 1200x960 px, uint8, EPSG:32719
[OK]   classmap.png → quicklook RGB exportado

Distribución de clases:
  Agua    (id=1):  12.3%  (141,696 px)
  Relave  (id=2):  34.1%  (392,832 px)
  Terreno (id=3):  53.6%  (617,472 px)
```

### Pipeline modular paso a paso

```bash
# Paso 1: Stack multibanda (inspección y depuración)
python -m satplatform.cli stack \
  --date 20240123 \
  -b B02=./B02.tif -b B03=./B03.tif -b B04=./B04.tif \
  --order B04 B03 B02

# Paso 2: Normalización por histograma
python -m satplatform.cli hist-norm \
  --date 20240123 \
  -b B03=./B03.tif -b B04=./B04.tif -b B08=./B08.tif \
  --out ./02-Work/HIST-NORM/20240123/hn.tif

# Paso 3: Clasificación sobre bandas normalizadas
python -m satplatform.cli classify \
  --date 20240123 \
  -b B03=./02-Work/HIST-NORM/20240123/B03_norm.tif \
  -b B04=./02-Work/HIST-NORM/20240123/B04_norm.tif \
  -b B08=./02-Work/HIST-NORM/20240123/B08_norm.tif \
  -b B11=./02-Work/HIST-NORM/20240123/B11_norm.tif \
  --out ./03-Products/CLASSMAP/20240123/classmap.tif \
  --png
```

---

## CLI — referencia de comandos

### `classify` — pipeline completo a classmap

```
python -m satplatform.cli classify [opciones]

Opciones requeridas:
  --date YYYYMMDD       Fecha de la imagen (usada en rutas de salida)
  -b B03=./path.tif     Par banda=ruta (repetible; mínimo B03, B04, B08)

Opciones opcionales:
  --roi   path.geojson  Región de interés para recorte (Feature/FeatureCollection/Geometry)
  --normalize           Aplica normalización percent-clip antes de clasificar
  --out   path.tif      Ruta de salida (si no, usa output_patterns['classmap'] de Settings)
  --png                 Exporta quicklook PNG junto al GeoTIFF
  --gdalwarp path       Ruta a gdalwarp si no está en PATH
  --root  path          Sobreescribe project_root de Settings
```

### `stack` — apila bandas en un GeoTIFF multibanda

```
python -m satplatform.cli stack --date YYYYMMDD -b BAND=path [--order B04 B03 B02]
```

### `hist-norm` — normaliza bandas por histograma

```
python -m satplatform.cli hist-norm --date YYYYMMDD
  [-b BAND=path ...]    Múltiples bandas: normaliza y stackea
  [-i path]             Una sola imagen: normaliza directamente
  [--out path]          Ruta de salida explícita
  [--order B04 B03 B02] Orden de bandas en el stack
```

---

## Índices espectrales disponibles

Calculados por `SpectralService.compute_indices()` vía `BandSet`.

### Disponibles hoy

| Índice | Fórmula | Bandas S2 | Uso en tranques |
|---|---|---|---|
| NDVI | (B08 - B04) / (B08 + B04) | B04, B08 | Cobertura vegetal, revegetación de taludes |
| NDWI | (B03 - B08) / (B03 + B08) | B03, B08 | Agua superficial, lixiviados |
| NDBI | (B11 - B08) / (B11 + B08) | B08, B11 | Superficies áridas, techos, depósitos |

### Planificados (Fase 5)

| Índice | Fórmula | Utilidad |
|---|---|---|
| BSI | ((B11+B04)-(B08+B02)) / ((B11+B04)+(B08+B02)) | Suelos desnudos y depósitos minerales — el más relevante para relaves |
| MNDWI | (B03 - B11) / (B03 + B11) | Mejor discriminación agua/suelo que NDWI clásico |
| SAVI | ((B08-B04)/(B08+B04+L)) * (1+L) donde L=0.5 | NDVI ajustado por suelo, útil en zonas áridas |
| EVI | 2.5*(B08-B04)/(B08+6*B04-7.5*B02+1) | Reducción de saturación atmosférica |

Para calcular índices desde Python:

```python
from satplatform.services.spectral_service import SpectralService
from satplatform.contracts.products import BandSet

svc = SpectralService()

# compute_indices devuelve GeoRaster multibanda (una banda por índice)
indices = svc.compute_indices(bandset, ["NDVI", "NDWI", "NDBI"])
# indices.data.shape → (3, H, W) — orden: NDVI, NDWI, NDBI
```

---

## Clases de cobertura

Las clases disponibles en el clasificador legacy corresponden a las tres macroclases del dominio de tranques:

| ID | Nombre | MacroClass | Color por defecto | Descripción |
|---|---|---|---|---|
| 1 | Agua | AGUA | Azul (#1F77B4) | Agua superficial, piscinas de decantación, lixiviados |
| 2 | Relave | RELAVE | Rojo (#D62728) | Material depositado activo o seco |
| 3 | Terreno | TERRENO | Verde (#98DF8A) | Suelo natural, vegetación, infraestructura circundante |

Las clases se definen en `00-Config/class_labels.json` y son completamente configurables. Para agregar subclases (ej. "Relave húmedo" / "Relave seco"):

```json
[
  {"id": 1, "name": "Agua",          "macro": "Agua",    "color": {"r": 31,  "g": 119, "b": 180}},
  {"id": 2, "name": "Relave húmedo", "macro": "Relave",  "color": {"r": 214, "g": 39,  "b": 40}},
  {"id": 3, "name": "Relave seco",   "macro": "Relave",  "color": {"r": 255, "g": 127, "b": 14}},
  {"id": 4, "name": "Terreno",       "macro": "Terreno", "color": {"r": 152, "g": 223, "b": 138}}
]
```

---

## Extender la plataforma

La arquitectura hexagonal garantiza que agregar un nuevo clasificador, lector o exportador no requiere modificar el dominio ni los services existentes.

### Agregar un clasificador basado en Random Forest

```python
# adapters/sklearn_pixel_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np

from ..contracts.core import ClassLabel
from ..contracts.geo import GeoRaster, GeoProfile
from ..contracts.products import BandSet

@dataclass(frozen=True)
class SKLearnPixelClassifier:
    """Adapter que implementa PixelClassifierPort con sklearn."""
    
    model_path: Path
    classes_def: Sequence[ClassLabel]
    band_order: tuple[str, ...]  # orden esperado por el modelo

    def name(self) -> str:
        return f"sklearn-{self.model_path.stem}"

    def classes(self) -> Sequence[ClassLabel]:
        return self.classes_def

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        model = joblib.load(self.model_path)
        
        # Construir matriz de features (N, F) desde BandSet
        arrays = [bands.bands[b].data.reshape(-1) for b in self.band_order]
        X = np.stack(arrays, axis=1).astype(np.float32)
        
        # Predecir
        y_pred = model.predict(X).astype(np.uint8)
        
        base = bands.bands[self.band_order[0]]
        h, w = base.data.shape
        profile = GeoProfile(
            count=1, dtype="uint8",
            width=w, height=h,
            transform=base.profile.transform,
            crs=base.profile.crs,
            nodata=0,
        )
        return GeoRaster(data=y_pred.reshape(h, w), profile=profile)
```

Conectarlo en `composition/di.py`:

```python
from satplatform.adapters.sklearn_pixel_classifier import SKLearnPixelClassifier
from satplatform.services.classmap_service import ClassMapService

def build_rf_classmap_service(settings: Settings) -> ClassMapService:
    reader  = GdalRasterReader()
    writer  = GdalRasterWriter()
    clipper = GdalWarpClipper(raster_reader=reader, raster_writer=writer)
    clf     = SKLearnPixelClassifier(
        model_path=Path("models/rf_v2.pkl"),
        classes_def=settings.classes,
        band_order=("B03", "B04", "B08", "B11"),
    )
    cmapper = LegacyClassMapAdapter()
    return ClassMapService(
        reader=reader, writer=writer, clipper=clipper,
        classifier=clf, cmapper=cmapper,
    )
```

El CLI y los tests no necesitan ningún cambio.

### Agregar un nuevo índice espectral

En `services/spectral_service.py`, agrega la entrada al diccionario `required`:

```python
required: dict[str, tuple[S2BandName, S2BandName]] = {
    "NDVI":  ("B08", "B04"),
    "NDBI":  ("B11", "B08"),
    "NDWI":  ("B03", "B08"),
    "BSI":   ("B11", "B04"),   # ← nuevo
    "MNDWI": ("B03", "B11"),   # ← nuevo
}
```

Para índices con más de dos bandas o fórmulas no normalizadas, agrega un método específico siguiendo el mismo patrón de `nd_index`.

### Agregar un nuevo exportador de reportes

Implementa `ReportExporterPort` en `adapters/`:

```python
# adapters/jinja_report_exporter.py
from jinja2 import Environment, FileSystemLoader

class JinjaReportExporter:
    """Genera reportes HTML desde plantillas Jinja2."""
    
    def render(self, template_id: str, context: dict, out_uri: str) -> str:
        env = Environment(loader=FileSystemLoader("templates/"))
        tmpl = env.get_template(f"{template_id}.html.j2")
        Path(out_uri).write_text(tmpl.render(**context), encoding="utf-8")
        return out_uri
```

---

## Tests

La suite está organizada en tres niveles con dependencias progresivas.

### Unit — sin datos reales, sin GDAL

Prueban contratos y servicios con arrays numpy sintéticos. Corren en cualquier entorno.

```bash
pytest tests/unit/ -q

# Con cobertura
pytest --cov=satplatform --cov-report=term-missing
```

Qué cubren:
- `contracts/`: `BandSet`, `GeoProfile`, `validate_profile_compat`, `SceneId`, `CRSRef`
- `services/`: `ClassMapService` con fakes de ports, `HistogramNormService`, `PreprocessingService.rgb_to_hsl`
- `composition/`: wiring de `di.build_*_service` valida que los puertos estén conectados
- `config.py`: validación de placeholders, parseo de YAML, `crs_out_ref()`
- **Regresiones**: 3 bugs históricos (SyntaxError, `bands.has()`, `BandSet` no importado) + 1 nuevo (NaN nodata en `validate_profile_compat`) + RGB→HSL con escalar común

### Integration — autogenera fixtures con `tifffile`

```bash
pytest tests/integration/ -m gdal -q
```

`test_gdal_raster_reader` genera su propio TIFF de 5×5 con `tifffile` y valida el fallback del reader.

### Cobertura actual

```
Módulo                                      Cobertura
contracts/core.py                           85%
contracts/geo.py                            58%
contracts/products.py                       78%
ports/*                                     100%
services/classmap_service.py                78%
services/histogram_norm_service.py          53%
services/preprocessing_service.py           61%
services/spectral_service.py                20%
services/training_service.py                38%
adapters/legacy_pixelclass_adapter.py       98%
adapters/legacy_histnorm_adapter.py         70%
adapters/legacy_classmap_adapter.py         56%
adapters/gdal_raster_reader.py              32%
composition/di.py                           91%
config.py                                   89%
─────────────────────────────────────────────────
TOTAL                                       62%
```

Total: **38 tests, 100% pass**. Próximos objetivos: `services/` ≥85%, `adapters/` ≥70%, `spectral_service.py` ≥80%.

### Calidad de código

```bash
# Linting
ruff check src/

# Type checking (estricto en contratos y ports)
mypy --strict src/satplatform/contracts/ src/satplatform/ports/
mypy src/satplatform/services/ src/satplatform/adapters/

# Formato
ruff format src/
```

---

## Decisiones de diseño

### ¿Por qué `argparse` y no Typer?

`cli.py` usa `argparse` estándar por portabilidad sin dependencias adicionales. Typer está en el roadmap (Fase 6) una vez que el pipeline esté estabilizado y los comandos no cambien de firma. Migrar de `argparse` a Typer es un refactor mecánico que no afecta la lógica de dominio.

### ¿Por qué `frozen=True` en todos los DTOs?

Los datos geoespaciales deben ser inmutables una vez construidos para garantizar reproducibilidad. Un `GeoRaster` que se modifica en un paso intermedio del pipeline invalida el linaje de datos sin generar error explícito. `frozen=True` convierte ese error silencioso en una excepción inmediata.

### ¿Por qué `validate_profile_compat` exige mismo dtype?

Previene mezclas silenciosas de `uint16` (Sentinel-2 raw) y `float32` (normalizado) en un `BandSet`. Si necesitas mezclar bandas de distintos dtypes, normaliza explícitamente todas antes de construir el `BandSet`. El pipeline correcto es: load → normalize all → BandSet → classify.

### ¿Por qué tres niveles de fallback en el reader?

Para que los tests unitarios corran en CI sin GDAL instalado. `tifffile` permite testear la lógica de dominio con arrays numpy puros sin dependencia de librerías geoespaciales. En producción siempre se usa rasterio o GDAL.

### ¿Por qué los adapters legacy existen?

Los adapters `legacy_*` encapsulan scripts pre-arquitectura que ya funcionaban en producción. Permiten que el pipeline corra mientras se desarrollan implementaciones más sólidas. Su nombre `legacy` es intencional: señalan que deben ser reemplazados, no que son erróneos.

### ¿Por qué `CRSRef` sin GDAL?

`contracts/geo.py` no puede importar GDAL porque los contratos deben ser puro Python para los tests unitarios. `CRSRef` hace comparación determinista mediante EPSG (si está disponible) o WKT normalizado (si no). La conversión a objetos GDAL/rasterio ocurre en los adapters, que sí pueden importar GDAL.

### ¿Por qué `composition/di.py` y no un framework DI?

Un framework de inyección de dependencias (FastAPI, inject, punq) agrega complejidad sin beneficio real para un pipeline de procesamiento de datos que se configura una vez por ejecución. `di.py` es código Python plano que se lee linealmente — más fácil de depurar y modificar que un contenedor DI automático.

---

## Limitaciones conocidas

### Técnicas

- **Sin reproyección automática:** todas las bandas deben estar en el mismo CRS y resolución antes de entrar al pipeline. Usa `gdalwarp` manualmente si alguna banda tiene CRS o pixel size distinto. El adapter `GdalWarpClipper` reprovecta al CRS del proyecto al hacer clip, pero no entre bandas.

- **Sin descarga de imágenes:** el pipeline asume que los JP2/TIF ya están en `01-Raw/s2/`. No integra con la API de Copernicus ni con STAC. La descarga es responsabilidad del operador.

- **Clasificador legacy = reglas simples:** el clasificador actual usa umbrales fijos basados en reflectancia (B03, B04, B08, B11). Para producción sobre tranques específicos se requiere un modelo entrenado con datos de campo validados. Los umbrales no son transferibles entre proyectos sin ajuste.

- **Sin series temporales:** cada ejecución procesa una fecha de forma independiente. No hay detección de cambio entre fechas, ni análisis de tendencias. Esta capacidad está planificada para Fase 7.

- **Sin procesamiento multi-proyecto en paralelo:** el pipeline procesa un proyecto a la vez. Múltiples tranques requieren múltiples ejecuciones secuenciales o un orquestador externo (pendiente Fase 8).

- **Sin cobertura de nubes automática:** el pipeline no evalúa el SCL (Scene Classification Layer) de Sentinel-2. Si la imagen tiene cobertura de nubes, el classmap tendrá errores silenciosos en las zonas cubiertas. Verificar manualmente antes de procesar.

### De datos

- **Sentinel-2 L2A únicamente:** el clasificador está calibrado para reflectancias de superficie (L2A). Imágenes L1C (TOA) producirán resultados incorrectos.

- **Resolución mínima 10m:** el `BandSet` valida que `resolution_m` coincida con el tamaño de pixel real. Bandas a 60m (B01, B09) no son compatibles con bandas a 10m en el mismo BandSet sin resampling previo.

---

## Contribuir

### Reglas del dominio (no negociables)

1. `services/` **nunca** importa de `adapters/` ni de `config.py`
2. `contracts/` **no tiene** dependencias externas excepto `pydantic`, `numpy` y stdlib
3. Todo nuevo service debe tener tests unitarios sin I/O antes de merge
4. Los ports **solo** definen `Protocol` — sin lógica de negocio
5. Los DTOs de entrada terminan en `*Spec`, los de salida en `*Result`
6. `frozen=True` en todos los `@dataclass` y modelos Pydantic de dominio

### Proceso de contribución

```bash
# 1. Fork y clonar
git clone https://github.com/TU_USUARIO/sat_platform.git

# 2. Crear rama
git checkout -b feature/sklearn-classifier

# 3. Instalar hooks pre-commit
pip install pre-commit
pre-commit install

# 4. Desarrollar con tests
pytest tests/unit/ -q              # debe pasar 100%
ruff check src/
mypy --strict src/satplatform/contracts/ src/satplatform/ports/

# 5. PR con descripción que incluya:
#    - qué problema resuelve
#    - qué tests se agregaron
#    - si modifica el pipeline, el flujo antes y después
```

### Convenciones de nombre

| Tipo | Convención | Ejemplo |
|---|---|---|
| Adapter | `<Backend><Rol>` | `GdalRasterReader`, `SKLearnPixelClassifier` |
| Service | `<Dominio>Service` | `ClassMapService`, `HistogramNormService` |
| Port | `<Rol>Port` | `RasterReaderPort`, `PixelClassifierPort` |
| DTO entrada | `<Acción>Spec` | `NormalizeSpec`, `ClassMapSpec` |
| DTO salida | `<Acción>Result` | `ClassMapResult`, `NormalizeManyResult` |
| Test unit | `test_<módulo>.py` | `test_geo.py`, `test_classmap_service.py` |

### Checklist antes de PR

- [ ] Tests unitarios nuevos para la lógica agregada
- [ ] `ruff check` sin errores
- [ ] `mypy` sin errores en `contracts/` y `ports/`
- [ ] Si cambia el CLI: actualizar sección de referencia en este README
- [ ] Si agrega un índice espectral: actualizar tabla de índices
- [ ] Si agrega un nuevo adapter: actualizar tabla de estado del proyecto

---

## Entorno reproducible

GDAL tiene dependencias de sistema difíciles de instalar consistentemente entre plataformas. Se recomiendan las siguientes opciones.

### Opción 1: conda (recomendado para usuarios)

```bash
conda create -n satplatform python=3.11 gdal rasterio numpy
conda activate satplatform
cd sat_platform
pip install -e ".[dev]"
python -m satplatform.cli --help
```

### Opción 2: pip + sistema (Linux/macOS)

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev python3-gdal

# macOS con Homebrew
brew install gdal

pip install GDAL==$(gdal-config --version) rasterio
pip install -e ".[dev]"
```

### Opción 3: Docker (próximamente)

```bash
# Pendiente — ver issue #XX
# docker build -t satplatform .
# docker run -v $(pwd)/Proyecto:/project satplatform classify \
#   --date 20240123 -b B03=/project/01-Raw/s2/B03.tif ...
```

El `Dockerfile` está planificado para Fase 8 junto con el soporte de CI/CD.

---

## Roadmap por fases

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Higiene — ruff + mypy + estructura base | ✅ Completa |
| 1 | `settings.yaml` válido, placeholders cerrados | ✅ Completa |
| 2 | Contracts inmutables (`GeoRaster`, `BandSet`, `ClassLabel`) | ✅ Completa |
| 3 | Ports cerrados y documentados | ✅ Completa |
| 4 | Services alineados con tests unitarios (dominio puro) | ✅ Completa — bugs bloqueantes corregidos, 38 tests |
| 5 | Adapters mínimos (GDAL reader/writer/clipper) | 🟡 Parcial — legacy funcional, falta sklearn classifier |
| 6 | CLI reproducible paso a paso (classify funcional) | ✅ Completa — entry point `satplatform`, wiring vía `di.py` |
| 7 | Exporters: reportes CSV, quicklooks PNG con metadatos | ⬜ Pendiente |
| 8 | CI/CD: tests automáticos + Docker + GitHub Actions | ⬜ Pendiente |

### Backlog posterior al Fase 8

- Detección de cambios entre fechas (diferencia de classmaps)
- Índices BSI, MNDWI, SAVI, EVI
- Integración con catálogo STAC de Copernicus para descarga automática
- Soporte multi-AOI y procesamiento por lotes
- Clasificador sklearn con entrenamiento desde datos de campo
- Validación con ground truth (métricas OA, Kappa, F1 por clase)
- Scheduler para ejecución automática al publicarse nuevas imágenes

---

## Referencias

- [ESA Sentinel-2 SAFE format specification](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats)
- [Sentinel-2 Level-2A Algorithm Theoretical Basis Document](https://sentinel.esa.int/documents/247904/685211/Sentinel-2_Level-2A_ATBD)
- [rasterio — Geospatial raster I/O for Python](https://rasterio.readthedocs.io/)
- [GDAL — Geospatial Data Abstraction Library](https://gdal.org/)
- [Pydantic v2 — Data validation](https://docs.pydantic.dev/latest/)
- Arquitectura hexagonal: Alistair Cockburn, "Hexagonal Architecture", 2005
- [MGRS — Military Grid Reference System](https://earth-info.nga.mil/index.php?dir=coordsys&action=mgrs)
- Tucker, C.J. (1979). "Red and photographic infrared linear combinations for monitoring vegetation" — origen de NDVI
- Zha, Y., Gao, J., Ni, S. (2003). "Use of normalized difference built-up index in automatically mapping urban areas" — origen de NDBI

---

## Licencia

MIT — libre uso y modificación con atribución.

```
Copyright (c) 2025 faltamiranod93

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
