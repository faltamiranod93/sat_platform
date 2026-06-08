# Última sesión — breadcrumb de trabajo

> Este archivo es actualizado por `/fin` al cerrar cada sesión.
> Viaja con git para que esté disponible en ambos computadores.
> `/inicio` lo lee para orientar el resumen de la próxima sesión.

---

## Última sesión activa
**Fecha:** 2026-06-08
**Computador:** universidad (geotecnia-usm)

## Qué se hizo
- `git pull` aplicado: 9 commits nuevos (Mahalanobis/Cosine/Euclidean + McalGeorefService + skills + GeoJSON UTM v7).
- Suite verde: **85/85 tests** (38 viejos + 47 nuevos para adapters nativos y mcal_georef). Los 4 tests que fallaban en personal ya están arreglados.
- Auditoría del Mcal completada: **11 clases reales** (no 3); v7 tiene 1721 muestras en 4 fechas; **552 puntos del 2024-01-23 coinciden con una de las 160 escenas** de `01-Laguna-Seca-example/`.
- Hallazgo crítico: el Mcal NO usa la misma grilla que los TIFFs Sentinel Hub (correlación banda-a-banda ~0). Solo coordenadas UTM + Ng son reutilizables.
- Output fuera de repo: `01-Laguna-Seca-example/04-Analysis/Mcal_audit.md` (no se commitea, son datos del Msc).

## Próximo paso inmediato
**Verificar que `McalHSL_mod_v7_py_utm.geojson` está realmente en EPSG:32719.** El usuario lo cuestionó al cerrar la sesión — sin esto, `extract_at_utm_points()` produce basura. Verificar bbox esperado para Laguna Seca: UTM 19S aprox. 485000-500000 E, 7290000-7305000 N. Si está mal proyectado, regenerar antes de continuar el Paso 3.

## Pendiente / bloqueado
- 🔴 **Bloqueador**: confirmar proyección del GeoJSON UTM v7 antes de extraer muestras.
- Paso 3 del plan post-pull pendiente: re-batch 160 escenas con Mahalanobis v9p2 + comparativa contra LegacyPixelClassifier.
- Falencias previas del clasificador legacy siguen sin atacarse: nodata=Agua, sin ROI de Laguna Seca, sin máscara de nubes.
- Migración `TemporalNormPort` + adapter (RRN multitemporal) seguía pendiente desde la sesión anterior.
