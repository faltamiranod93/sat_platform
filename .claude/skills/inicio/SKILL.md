---
name: inicio
description: Resumen de sesión al iniciar trabajo en MSc-UTFSM. Sincroniza git, lee SESION.md y memorias, y presenta un resumen orientador de las líneas de trabajo (sat-platform, Laguna Seca, literatura). Invocar con /inicio al empezar a trabajar.
---

# /inicio — Resumen de sesión al iniciar trabajo en MSc-UTFSM

Cuando el usuario invoca `/inicio`, ejecuta los siguientes pasos **en orden**:

## Paso 1 — Sincronizar git

Corre en `sat-platform/`:
```
git pull origin main
git status
git log --oneline -8
```

Informa brevemente:
- Si llegaron cambios nuevos: qué archivos cambiaron y desde qué commits
- Si ya estaba al día: confirmarlo en una línea
- Si hay cambios locales sin commitear: listarlos con advertencia

## Paso 2 — Leer estado del proyecto

Lee en este orden:
1. `sat-platform/.claude/ATLAS.md` — el **Atlas** (mapa estratégico de las 3 aristas). Da el marco: dónde va la tesis y qué frente toca cada arista.
2. `sat-platform/.claude/SESION.md` — breadcrumb de la última sesión (qué se hizo, próximo paso, bloqueados)
3. `MEMORY.md` (índice de memorias locales)
4. Cualquier archivo de memoria relevante que el índice señale

El Atlas da la visión estratégica; SESION.md tiene prioridad para el "pendiente prioritario" — es lo más fresco.

## Paso 3 — Presentar resumen por líneas de trabajo

Muestra un resumen estructurado con este formato exacto. Encabézalo con el panorama por **aristas** (del Atlas) y luego el detalle por línea:

---
### Estado del proyecto MSc-UTFSM — [fecha actual]

**Aristas (Atlas):** 1) Completar desarrollos · 2) Validar resultados · 3) Plan tesis — [1 línea con en qué arista está el foco ahora]

**sat-platform**
[1–3 líneas sobre el último estado de la plataforma: qué adapters/services existen, qué está pendiente]

**Laguna Seca**
[1–2 líneas sobre las escenas disponibles, último procesamiento conocido, qué falta correr]

**Literatura / Papers**
[1 línea indicando que hay INDEX.md en `10 papers/` con los 6 papers de RRN]

**Pendiente prioritario**
[El ítem más importante de continuar según el contexto acumulado]

---

## Paso 4 — Preguntar en qué continuar

Termina con exactamente esta pregunta:

> ¿En qué línea de trabajo continuamos hoy?
> 1. sat-platform (desarrollo de la plataforma)
> 2. Laguna Seca (procesamiento / clasificación de escenas)
> 3. Literatura / papers
> 4. Otro (dime)

Espera la respuesta del usuario antes de hacer cualquier otra cosa.

## Notas
- Si el usuario está en el computador de la universidad, las rutas serán distintas. Adaptarse según el working directory actual.
- Si hay errores en el git pull (conflictos, credenciales), reportarlos claramente antes de continuar con el resumen.
- El resumen debe ser **conciso**: máximo 15 líneas en total. No es un informe — es un orientador de sesión.
