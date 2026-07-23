---
name: project-atlas-msc
description: Atlas / mapa estratégico del Msc-sentinel2 — las 3 aristas (completar desarrollos, validar resultados, plan tesis) y qué memoria/artefacto alimenta cada una
metadata:
  type: project
---

# Atlas MSc-sentinel2

> Copia **git-trackeada** del atlas (viaja entre computadores). Espejo humano en
> `md-faltamirano/ATLAS.md` y memoria oficial `project-atlas-msc`. Las tres se mantienen
> idénticas con la skill `/atlas`.

Mapa **estratégico** de todo `~/Documents/Msc-sentinel2/`, organizado en **tres aristas**. Es el punto de entrada: dice *dónde va la tesis* y a qué memoria/artefacto ir para cada frente.

**Cómo navegar (capas):**
- **Atlas** (este archivo) = estratégico, "dónde va la tesis" (visión por aristas + roadmap).
- **`sat_platform/.claude/SESION.md`** = táctico, "qué hice la última sesión / próximo paso" (git-trackeado, viaja entre computadores). Es la fuente más fresca del estado de la Arista 1.
- **`geodata/`** = sub-proyecto **cerrado** (curso INF491, defensa 1-jul-2026); antecedente metodológico, no trabajo activo. Ver [[project-geodata-inf491]].
- Skills de sesión ya existentes: `/inicio` (abre sesión), `/fin` (cierra), `/sync` (git). El atlas se apoya encima de ellas, no las reemplaza.

---

## Arista 1 — Plataforma y procesamiento
**Qué es:** terminar la plataforma `sat_platform` y el procesamiento de Laguna Seca.
**Enlaces:** [[project-sat-platform]] · `sat_platform/.claude/SESION.md` (estado vigente) · [[laguna-eval-m1-baseline]] · [[laguna-mcal-v7-taxonomy]] · [[laguna-mcal-grid-mismatch]]

**Estado actual** (2026-07-22 — ver `sat_platform/.claude/SESION.md`):
- Arquitectura hexagonal estable. **Milestone 1 (validación) implementado y ejecutado**: módulo de evaluación con CV espacial por bloques + TFC temporal (`evaluation_service`/`_folds`/`_metrics`), puerto con `predict_points`, subcomando CLI `evaluate`. Suite **186 tests verde**.
- **Primer baseline honesto** (Laguna Seca 2024-01-23): la CV aleatoria inflaba la accuracy → gap OA +0.08, kappa 0.87→0.35 con bloqueo espacial; confusión dominante clase 3↔11. Ver [[laguna-eval-m1-baseline]].

**Próximos pasos:**
- Correr **P2/P3 (TFC temporal)** con escenas 2020/21/22 (faltan en el ejemplo; existen en el GeoJSON v7).
- **M2: portar el RRN legacy** a `TemporalNormPort/Adapter/Service` y medirlo con el hook `norm=` del evaluador (no-regresión à la Gan). Estrategia de ajuste diferida (legacy fiel / polígonos Chen / MSAC+Tukey Xu).
- Separabilidad **clase 3 vs 11** sigue pendiente, ahora cuantificada por el evaluador. Materializar FEATURES/MASK.

---

## Arista 2 — Contraste satélite–terreno
**Qué es:** contrastar la salida satelital (superficie NDWI, classmaps Mahalanobis) contra datos de terreno reales.
**Enlaces:** [[project-laguna-seca-terreno]] (monitoreo Geomonitoring/BHP: batimetría AUV + clasificación GeoEye 7 clases + planillas TLS) · [[project-laguna-seca-findings]] ("Agua"=nodata, umbrales sin calibrar) · [[laguna-mcal-grid-mismatch]] (todo en UTM 19S)

**Estado actual:**
- Disponible el **monitoreo profesional de terreno** en `info Laguna-Seca/`: informes Geomonitoring (`Informes/`) + 5 planillas operacionales (`TLS/`). El informe EB-03 (2019) es ground truth **y benchmark**: hace monitoreo satelital + clasificación supervisada de 7 clases sobre GeoEye-1 (50 cm) — justo lo que la tesis replica con Sentinel-2 (10 m).
- Aún **sin validación cuantitativa**: los 552 pts de entrenamiento no tienen split train/test; los classmaps no están contrastados con terreno.

**Próximos pasos:**
- Cruzar el **"Espejo de Agua"** batimétrico (enero 2024) con la superficie NDWI de la escena satelital **2024-01-23**.
- Contrastar volumen/cota de laguna vs. área de agua detectada; recalibrar umbrales (arreglar H1 "Agua"=nodata primero).
- **Comparar la clasificación 11-clases Sentinel-2 vs. las 7 clases GeoEye de Geomonitoring** (mapeo de clases, superficies por zona).
- Validación cuantitativa del clasificador (matriz de confusión con los 552 pts).

---

## Arista 3 — Documento de tesis
**Qué es:** roadmap para escribir y defender la tesis de magíster. **Parte desde cero** (no existe documento de tesis todavía).
**Enlaces:** [[project-geodata-inf491]] (antecedente metodológico: GWR/LISA/SAR ya aplicados y defendidos) · literatura RRN/papers (por materializar; la skill `/inicio` referencia un `10 papers/` que aún no existe aquí).

**Roadmap (desde cero):**
- [ ] Pregunta de investigación / objetivos (general + específicos)
- [ ] Marco teórico + revisión de literatura (RRN, NDWI/NDMI, econometría espacial, monitoreo de relaves)
- [ ] Metodología → apóyate en `sat_platform` (Arista 1)
- [ ] Resultados → validados contra terreno (Arista 2)
- [ ] Escritura por capítulos
- [ ] Revisión / correcciones
- [ ] Defensa

**Por definir (huecos a rellenar):** plazos del programa · formato/plantilla exigida · comité/profesor guía.

---

## Historial
_(la skill `/atlas` agrega aquí una entrada por sesión relevante; nunca borra entradas)_

### 2026-07-22
- **Arista 1 — Milestone 1 (validación) implementado y ejecutado.** Módulo de evaluación (CV espacial por bloques + TFC temporal), refactor del puerto con `predict_points`, CLI `evaluate`; **186 tests verde**. Primer baseline honesto en Laguna Seca 2024-01-23: la CV aleatoria inflaba (gap OA +0.08, kappa 0.87→0.35), confusión dominante clase 3↔11 (ver [[laguna-eval-m1-baseline]]). Precedido por el estudio a fondo de 9 papers en `literatura/INDEX.md` + roadmap A–E de la arista.
- **Renombradas las 3 aristas** para reflejar mejor lo que se hace: Arista 1 → *Plataforma y procesamiento*, Arista 2 → *Contraste satélite–terreno*, Arista 3 → *Documento de tesis*. Solo nomenclatura; sin cambios de alcance. Sincronizado en las 3 copias + `CLAUDE.md`.

### 2026-07-14
- **Arista 2 reforzada:** documentado el monitoreo profesional de terreno (Geomonitoring/BHP) en `info Laguna-Seca/`. Hallazgo clave: el informe EB-03 (2019) hace clasificación satelital supervisada de 7 clases sobre GeoEye-1 (50 cm) → es el **benchmark comercial** de la tesis (Sentinel-2, 10 m). Reorganizada `Informes/` por fecha/servicio. Memoria [[project-laguna-seca-terreno]] ampliada.

### 2026-07-09
- **Sistema Atlas creado y operativo.** Definidas las 3 aristas y mapeadas a memorias/artefactos. Creadas memorias `project-geodata-inf491` y `project-laguna-seca-terreno` (carpetas nuevas: curso INF491 + datos de terreno). `CLAUDE.md` en la raíz, skill `/atlas`, y enganche en `/inicio` y `/fin`. Arreglados symlinks rotos de skills en `~/.claude/skills` (ver [[reference-skills-symlinks]]). Explorador Streamlit confirmado commiteado (`d8c5ad8`). Arista 3 queda como roadmap vacío: **pendiente que el usuario aporte plazos/formato/comité del programa**.
