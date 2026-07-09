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
- **`sat_platform/.claude/SESION.md`** = táctico, "qué hice la última sesión / próximo paso". Es la fuente más fresca del estado de la Arista 1.
- **`geodata/`** = sub-proyecto **cerrado** (curso INF491, defensa 1-jul-2026); antecedente metodológico, no trabajo activo. Ver [[project-geodata-inf491]].
- Skills de sesión: `/inicio` (abre), `/fin` (cierra), `/sync` (git). El atlas se apoya encima de ellas.

---

## Arista 1 — Completar los desarrollos
**Qué es:** terminar la plataforma `sat_platform` y el procesamiento de Laguna Seca.
**Enlaces:** [[project-sat-platform]] · `sat_platform/.claude/SESION.md` (estado vigente) · [[laguna-mcal-v7-taxonomy]] · [[laguna-mcal-grid-mismatch]]

**Estado actual** (según SESION.md, 2026-06-10 — más fresco que la memoria de sat_platform):
- Arquitectura hexagonal estable; `classify-batch` al esquema de carpetas estándar; **231 fechas clasificadas**; explorador de firmas Streamlit (firmas/separabilidad/visor). Suite **151 tests verde**.

**Próximos pasos:**
- Mejorar separabilidad **clase 3 (terreno natural) vs 11 (sombreado)** — Mahalanobis 71% en sombreado (más índices / training multitemporal / redefinir clases); usar el explorador Streamlit (ya commiteado, `d8c5ad8`) para decidir cómo.
- Materializar FEATURES/MASK (reservados en el contrato).
- Pendientes de plataforma (memoria vieja, reverificar): cobertura de `spectral_service.py`, Fase 5 (BSI/MNDWI/SAVI/EVI), Fase 7 (exporters), Fase 8 (CI/CD).

---

## Arista 2 — Validar los resultados
**Qué es:** contrastar la salida satelital (superficie NDWI, classmaps Mahalanobis) contra datos de terreno reales.
**Enlaces:** [[project-laguna-seca-terreno]] (5 Excel operacionales nuevos) · [[project-laguna-seca-findings]] ("Agua"=nodata, sin filtro de calidad, umbrales sin calibrar)

**Estado actual:**
- Llegaron datos de terreno (`Laguna-Seca Info/TLS/`): volumen/cota de laguna, batimetría (áreas por zona, incl. "Espejo de Agua"), revancha por PK, depositación por sector.
- Aún **sin validación cuantitativa**: los 552 pts de entrenamiento no tienen split train/test; los classmaps no están contrastados con terreno.

**Próximos pasos:**
- Cruzar el **"Espejo de Agua"** batimétrico (enero 2024) con la superficie NDWI de la escena satelital **2024-01-23** (fechas más cercanas).
- Contrastar volumen/cota de laguna vs. área de agua detectada; usar como ground truth para recalibrar umbrales (arreglar H1 "Agua"=nodata primero).
- Validación cuantitativa del clasificador (matriz de confusión con los 552 pts).

---

## Arista 3 — Plan para finalizar la tesis
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

### 2026-07-09
- **Sistema Atlas creado y operativo.** Definidas las 3 aristas y mapeadas a memorias/artefactos. Creadas memorias `project-geodata-inf491` y `project-laguna-seca-terreno` (carpetas nuevas: curso INF491 + datos de terreno). `CLAUDE.md` en la raíz, skill `/atlas`, y enganche en `/inicio` y `/fin`. Arreglados symlinks rotos de skills en `~/.claude/skills` (ver [[reference-skills-symlinks]]). Explorador Streamlit confirmado commiteado (`d8c5ad8`). Arista 3 queda como roadmap vacío: **pendiente que el usuario aporte plazos/formato/comité del programa**.
