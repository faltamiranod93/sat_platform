# s_sat
"Repository created for multispectral satellital imagery investigation"
# sat-platform

**sat-platform** es una plataforma modular para procesar imágenes multiespectrales de Sentinel-2 aplicada a tranques de relaves y monitoreo ambiental.  
El diseño sigue **arquitectura hexagonal (ports & adapters)**: el dominio (servicios + contratos) no depende de frameworks ni librerías externas. Las implementaciones concretas (GDAL, exportadores, legacy scripts) se encapsulan en *adapters*.

---

## 📂 Estructura del proyecto
```
sat-platform/
├─ pyproject.toml # dependencias y build
├─ README.md # este archivo
├─ src/satplatform/ # código fuente
│ ├─ contracts/ # DTOs de dominio (GeoRaster, BandSet, etc.)
│ ├─ ports/ # Protocolos (interfaces) de entrada/salida
│ ├─ services/ # lógica de dominio pura
│ ├─ adapters/ # adapters concretos (GDAL, legacy, exportadores)
│ ├─ composition/di.py # dependency injection: wiring puertos→adapters
│ ├─ config.py # Settings y validación de placeholders
│ └─ cli.py # CLI (Typer) con comandos reproducibles
├─ tests/ # suite de tests
│ ├─ unit/ # unit tests (dominio puro, sin I/O)
│ ├─ integration/ # integración (requiere GDAL/datos pequeños)
│ └─ e2e/ # end-to-end (CLI con Typer runner)
└─ 00-Config/, 01-Raw/, ... # estructura física de proyectos (ver abajo)
```
## 📂 Estructura de proyectos (layout físico)

Un proyecto se organiza así:
```
Proyecto/
├─ 00-Config/
│ ├─ settings.yaml # configuración de paths, CRS, patrones
│ ├─ class_labels.json # catálogo de clases (id, nombre, color RGB)
│ └─ roi_master.geojson # AOIs maestras
├─ 01-Raw/ # datos crudos (intocables)
│ ├─ s2/ # productos SAFE de Sentinel-2
│ │ └─ S2A_MSIL2A_20240123...SAFE/...
│ ├─ DEM/ # modelos digitales de elevación
│ └─ Ancillary/ # capas auxiliares
├─ 02-Work/ # intermedios (recortes, stacks, features)
│ ├─ ROI/
│ ├─ STACK/
│ ├─ HIST-NORM/
│ ├─ FEATURES/
│ └─ ...
├─ 03-Products/ # productos finales
│ ├─ CLASSMAP/
│ ├─ CLASS-VIS/
│ ├─ VIS/
│ ├─ VIS-MOD/
│ └─ REPORT/
└─ 04-Analysis/ # notebooks, scripts, figuras
```

---

## ⚙️ Configuración (Settings)

### `00-Config/settings.yaml`
Ejemplo:

```yaml
project_root: "."
crs_out: "EPSG:32719"

work_roi_dir: "02-Work/ROI"
work_products_dir: "03-Products"
report_dir: "03-Products/REPORT"

gdalwarp_exe: null

band_order: ["B02","B03","B04"]
classes: []

input_patterns:
  safe_dir: "01-Raw/s2/{product}.SAFE"
  granule_dir: "01-Raw/s2/{product}.SAFE/GRANULE/{granule}"
  jp2_file: "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_{band}_{res}.jp2"
  scl_file: "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_SCL_{res}.jp2"
  mask_file: "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/QI_DATA/MSK_QUALIT_{band}.jp2"
  roi_file: "00-Config/roi_master.geojson"

output_patterns:
  stack: "02-Work/STACK/{date}/stack.tif"
  hist_norm: "02-Work/HIST-NORM/{date}/hn.tif"
  features_hsl: "02-Work/FEATURES/{date}/hsl.tif"
  classmap: "03-Products/CLASSMAP/{date}/classmap.tif"
👉 Placeholders permitidos:

SAFE: {product}, {granule}, {tile}, {sensing}, {band}, {res}

Work/Products: {date}
```

🧩 Arquitectura hexagonal
Contracts: tipos de dominio (GeoRaster, BandSet, ClassLabel, S2Asset).

Ports: interfaces (RasterReaderPort, ClassMapPort, PixelClassifierPort, etc.).

Services: lógica pura (histogram normalization, preprocessing, spectral indices, training).

Adapters: implementaciones concretas (GDAL, CSV, legacy).

Composition: conecta puertos ↔ adapters en di.py.

CLI: expone comandos reproducibles.

🚀 CLI (Typer)
Ejemplos:

# Recortar ROI
sat-platform roi clip --roi-id ROI1 --date 20240123

# Construir stack RGB
sat-platform stack build --date 20240123 --order B02,B03,B04

# Normalización por histograma
sat-platform hist-norm run --date 20240123

# Features HSL
sat-platform features hsl --date 20240123

# Generar classmap
sat-platform classmap run --date 20240123 --model baseline

# Exportar quicklook
sat-platform export quicklook --date 20240123
Todos los artefactos quedan en 02-Work/ o 03-Products/ siguiendo los patrones de settings.yaml.

🧪 Tests
La suite está organizada en 3 niveles:

Unit (tests/unit/): dominio puro, sin I/O. → debe correr en cualquier entorno.

Integration (tests/integration/): adapters reales (GDAL, legacy). → requiere GDAL instalado.

E2E (tests/e2e/): ejecución del CLI con datos sintéticos.

Ejemplo:

```
pytest tests/unit -q          # solo dominio
pytest tests/integration -m gdal   # solo si tienes GDAL
pytest tests/e2e -q
```

🧹 Calidad de código
Linters: ruff y mypy --strict.

Pre-commit hooks: .pre-commit-config.yaml en la raíz asegura que no se commitea código roto.

Instalación:

```
pip install pre-commit ruff mypy
pre-commit install
```
📌 Roadmap / fases
Fase 0: higiene → ruff + mypy + pytest unit 100% verde.

Fase 1: settings.yaml válido y placeholders cerrados.

Fase 2: contracts inmutables (GeoRaster, BandSet).

Fase 3: cierre de firmas de ports.

Fase 4: services alineados con tests (dominio puro).

Fase 5: adapters mínimos (GDAL reader/writer).

Fase 6: CLI reproducible paso a paso (ROI → stack → hist-norm → features → classmap → export).

Fase 7: exporters/reportes.

Fase 8: CI/CD (unit siempre; integración opcional).

📖 Referencias
ESA Sentinel-2 SAFE format specification

GDAL & rasterio for geospatial I/O

Arquitectura hexagonal (ports & adapters) aplicada a procesamiento satelital

📝 Licencia
MIT (libre uso y modificación, con atribución).
