# Explorador de firmas espectrales (Streamlit)

Herramienta interactiva que reemplaza los `legacy/scripts/g_Sat_SpectralSignature_v*`.
Reutiliza el pipeline: re-extrae las firmas de las **escenas reales** (GeoJSON v7 +
`extract_at_utm_points`, georef corregida al vuelo), calcula **bandas + HSL + índices**
con `FeatureService`, y permite analizar separabilidad de las clases.

## Instalar y lanzar

```bash
pip install -e '.[viz]'      # streamlit, plotly, streamlit-image-coordinates

streamlit run tools/spectral_explorer/app.py -- \
  --root   /home/.../01-Laguna-Seca-example \
  --geojson /home/.../sat_platform/legacy/data/Laguna-Seca/McalHSL_mod_v7_py_utm.geojson \
  --scenes-glob "01-Raw/*/S2*MSIL2A*"
```
(Los argumentos van tras `--`; también pueden editarse en la barra lateral.)

## Vistas

- **📈 Firmas de clases**: media±std de cada feature (bandas/HSL/índices) por clase,
  superponibles para comparar. Color por clase desde `class_labels.json`.
- **🔀 Separabilidad**: matriz de distancia entre clases (euclidiana / Mahalanobis) —
  valores bajos fuera de la diagonal = clases que se confunden — y proyección PCA 2D.
- **🗺️ Visor de escena**: imagen RGB (TrueColor/FalseColor/SWIR) por fecha, con los
  puntos del GeoJSON superpuestos; clic en un píxel → su firma espectral.

## Arquitectura

La lógica es **dominio puro y testeada**; Streamlit es solo presentación:
- `services/spectral_signature_service.py` — `signatures_by_class`, `separability_matrix`, `pca_2d`, `temporal_means`.
- `services/scene_view_service.py` — `rgb_composite`, `points_to_pixels`, `pixel_signature`.
- Reutiliza `FeatureService`, `multiband_loader`, `di.build_training_set`, `di.build_raster_reader(fix_georef=True)`.

## Para qué sirve en el proyecto

Diagnostica la **separabilidad de las 11 clases** con las features actuales: si dos
clases (p.ej. los relaves vs "terreno sombreado") quedan muy cerca en la matriz/PCA,
explica el sesgo de los clasificadores observado en el batch.
