# Esquema de carpetas del pipeline (contrato de rutas)

Cada etapa escribe en su carpeta, de forma **trazable por `{date}`** (YYYYMMDD).
La fuente de verdad es `Settings.output_patterns` (`src/satplatform/config.py`),
validada contra `OUTPUT_PLACEHOLDERS`. El `00-Config/settings.yaml` del proyecto
puede sobrescribir las rutas; los servicios nunca las hardcodean (reciben
*resolvers* inyectados desde el composition root).

## Estructura (esquema plano)

```
<project_root>/
├── 00-Config/                 settings.yaml, class_labels.json, roi_master.geojson
├── 01-Raw/                    escenas descargadas (multibanda, ya recortadas a ROI y stackeadas)
├── 02-Work/                   derivados intermedios reproducibles
│   └── FEATURES/{date}/       índices/HSL materializados   [reservado, opcional]
│   └── MASK/{date}/           máscara de nubes/sombra       [reservado, futuro]
├── 03-Products/               salidas finales
│   ├── CLASSMAP/{date}/classmap_{classifier}.tif
│   └── VIS/{date}/classmap_{classifier}.png
└── 04-Analysis/               análisis agregado / comparativas
    └── CLASSMAP-COMPARE/      counts.csv, agreement.csv
```

## Contrato (`output_patterns`)

| key | patrón | placeholders | escribe |
|-----|--------|--------------|---------|
| `classmap` | `03-Products/CLASSMAP/{date}/classmap_{classifier}.tif` | `date`, `classifier` | `classify`, `classify-batch` |
| `classmap_vis` | `03-Products/VIS/{date}/classmap_{classifier}.png` | `date`, `classifier` | `classify-batch` |
| `features` | `02-Work/FEATURES/{date}/features.tif` | `date` | (reservado) |
| `compare_summary` | `04-Analysis/CLASSMAP-COMPARE/{name}.csv` | `name` | `classify-batch` |

## Notas de diseño

- **STACK / HIST-NORM** salieron del flujo: la descarga (`s_sen2_down_v3.py`)
  ya entrega un GeoTIFF multibanda recortado al ROI, así que apilar bandas
  sueltas no aplica. Los comandos CLI `stack`/`hist-norm` se conservan para
  casos legacy, pero **exigen `--out` explícito** (fuera del contrato).
- **ROI** se conserva para refinar zonas puntuales dentro del AOI descargado;
  no se usa por defecto.
- **Una salida por fecha**: `classify-batch` deduplica las escenas por fecha
  (`scene_index_from_uris`), evitando colisiones con `{date}`.
- Las escenas Sentinel Hub se guardan **sin extensión**; los sidecars
  (`.aux.xml`, etc.) se filtran con `is_scene_file`.
