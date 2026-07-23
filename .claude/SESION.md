# Última sesión — breadcrumb de trabajo

> Este archivo es actualizado por `/fin` al cerrar cada sesión.
> Viaja con git para que esté disponible en ambos computadores.
> `/inicio` lo lee para orientar el resumen de la próxima sesión.

---

## Última sesión activa
**Fecha:** 2026-07-22
**Computador:** universidad (geotecnia-usm)

## Qué se hizo
- **Milestone 1 (Arista 1) implementado: módulo de evaluación.** Refactor del puerto `PixelClassifierPort` (nuevo `predict_points` + `_score` compartido en los 3 clasificadores; legacy lanza `NotImplementedError`). Nuevos servicios puros `evaluation_folds.py` (spatial-block CV / leave-one-date-out / anchor), `evaluation_metrics.py` (OA/Kappa/PA-UA/F1/Pontius + Moran's I) y `evaluation_service.py` (4 protocolos P0–P3, exclusión de clases <2, hook `norm=` para el RRN). Wiring en `di.py` + subcomando CLI `evaluate`. **35 tests nuevos, 186 en total, todo verde.**
- **Ejecutado en `01-Laguna-Seca-example`** (2024-01-23, 552 pts, 11 clases): CV aleatoria infla la métrica → **gap OA P0−P1 ≈ +0.08, kappa 0.87→0.35**; confusión dominante clase 3↔11. CSVs en `04-Analysis/EVAL/`. Ver [[laguna-eval-m1-baseline]].
- Antes: estudio a fondo de 9 papers en `literatura/INDEX.md` + roadmap A–E de Arista 1 (spec del M1 al 100%). Aristas renombradas.

## Próximo paso inmediato
Correr P2/P3 (TFC temporal) con escenas 2020/21/22, luego arrancar **M2: portar el RRN legacy** a `TemporalNormPort/Adapter/Service` y medirlo con el hook `norm=` del evaluador (test de no-regresión).

## Pendiente / bloqueado
- **P2/P3 en el ejemplo:** el dir `01-Laguna-Seca-example` solo tiene escenas 2023/2024/2025; el TFC temporal necesita las escenas 2020-04-03 / 2021-04-28 / 2022-04-28 (existen en el GeoJSON v7, faltan las escenas).
- **M2 (RRN):** estrategia de ajuste diferida (portar legacy fiel vs polígonos Chen vs MSAC+Tukey Xu) — decidir al implementar.
- `literatura/INDEX.md` vive fuera de git (no viaja); espejar si se necesita en el otro computador.
