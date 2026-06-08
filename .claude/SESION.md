# Última sesión — breadcrumb de trabajo

> Este archivo es actualizado por `/fin` al cerrar cada sesión.
> Viaja con git para que esté disponible en ambos computadores.
> `/inicio` lo lee para orientar el resumen de la próxima sesión.

---

## Última sesión activa
**Fecha:** 2026-06-08
**Computador:** personal (C:/Users/felip)

## Qué se hizo
- `McalGeorefService.to_geojson()`: exporta ground truth como GeoJSON puro (geometría + Ng + Fecha, sin bandas espectrales). 19 tests pasando.
- `McalHSL_mod_v7_py_utm.geojson` generado y commiteado (1 721 puntos, EPSG:32719). Abre en QGIS con CRS 32719.
- Skills `/inicio` y `/fin` creados y pusheados al repo.
- `MEMORY.md` reorganizado por secciones (sat-platform / Laguna Seca / Workflow).

## Próximo paso inmediato
Llevar el trabajo al computador de la universidad:
1. `git pull` para traer todo lo hecho
2. Adaptar rutas via `MSC_UTFSM_ROOT` env var (ver `legacy/env_config.py`)
3. Usar `McalGeorefService.extract_at_utm_points()` para re-extraer valores espectrales en el nuevo ROI universitario
4. Generar nuevo Mcal desde los puntos UTM + imágenes de la universidad

## Pendiente / bloqueado
- 4 tests pre-existentes fallando (no causados por trabajo reciente): `test_settings_parse_and_paths`, `test_classmap_service_contracts`, `test_rgb_to_hsl_contract_and_ranges`, `test_discover_and_build_row`
- Migración `TemporalNormPort` + adapter (RRN multitemporal) pendiente para después de la universidad
- Prueba end-to-end Mahalanobis sobre escena real de Laguna-Seca pendiente
