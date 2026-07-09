---
name: atlas
description: Actualiza el Atlas MSc-sentinel2 (mapa estratégico de las 3 aristas — completar desarrollos, validar resultados, plan tesis). Reescribe el "Estado actual" de cada arista y agrega una entrada al historial. Úsala al cerrar trabajo relevante, cuando el usuario pida actualizar el atlas, o para ver el panorama global del Msc.
---

# /atlas — Atlas estratégico del MSc-sentinel2

Mantiene el mapa **estratégico** de toda la tesis en tres copias sincronizadas. El atlas es la capa "dónde va la tesis"; el detalle táctico de la última sesión vive en `SESION.md` (lo maneja `/fin`), no aquí.

## Las 3 copias del atlas (mantener idénticas)
1. **Memoria oficial:** `~/.claude/projects/-home-geotecnia-usm/memory/project_atlas_msc.md`
2. **Espejo humano:** `~/Documents/Msc-sentinel2/md-faltamirano/ATLAS.md`
3. **Copia git-trackeada (viaja entre computadores):** `~/Documents/Msc-sentinel2/sat_platform/.claude/ATLAS.md`

> Las rutas absolutas cambian según el computador — resuélvelas desde el working directory actual (`sat_platform/` es la ancla git).

## Qué hacer al invocarse

1. **Leer el estado real, sin volcar contenidos largos:**
   - Las 3 copias del atlas (si existen) para no contradecir el historial previo.
   - `sat_platform/.claude/SESION.md` — **fuente de verdad de la Arista 1** (lo más fresco).
   - `MEMORY.md` (índice de memorias) y las memorias que alimentan cada arista:
     - Arista 1: `project-sat-platform`, `laguna-mcal-v7-taxonomy`, `laguna-mcal-grid-mismatch`.
     - Arista 2: `project-laguna-seca-terreno`, `project-laguna-seca-findings`.
     - Arista 3: `project-geodata-inf491` (antecedente), literatura/papers.
   - Estado real en disco de lo tocado esta sesión (archivos nuevos/modificados; en `sat_platform/` puedes usar `git status`/`git log --oneline -5`).

2. **Comparar atlas vs. realidad.** Si una arista avanzó (hito cerrado, dato nuevo, decisión tomada), actualízala. Si un dato pertenece a una memoria concreta (no al atlas), corrige esa memoria y su línea en `MEMORY.md` — el atlas enlaza, no absorbe.

3. **Reescribir el atlas** conservando su estructura exacta (cabecera "Cómo navegar" + las 3 secciones de arista + historial):
   - En cada arista, **sobrescribe** "Estado actual" y "Próximos pasos" (son la foto del ahora).
   - En la **Arista 3**, marca `[x]` los ítems del roadmap cumplidos; rellena los huecos (plazos/formato/comité) si el usuario los aportó.
   - En **Historial**, agrega **una entrada nueva al inicio del historial** con la fecha de hoy (`currentDate`); nunca borres entradas. Si ya hay una de hoy, complétala.

4. **Sincronizar las 3 copias:** deja el mismo contenido en las tres. La copia de `md-faltamirano/` y la de `sat_platform/.claude/` conservan el frontmatter YAML igual que la memoria oficial.

5. **Fecha:** usa `currentDate` del contexto. No inventes fechas ni uses relativas.

## Estilo
- Conciso y en español. Viñetas, no párrafos.
- El "Estado actual" de cada arista debe leerse en pocos segundos y dejar claro dónde retomar.
- No copies contenido extenso de notebooks/rasters/PDFs; resume y enlaza.

## Relación con las otras skills
- `/inicio` lee el atlas como resumen estratégico antes del detalle por línea.
- `/fin` actualiza `SESION.md` (táctico) y, si la sesión movió un hito de arista, invoca la lógica de `/atlas`.
- `/sync` es solo git; no toca el atlas.
