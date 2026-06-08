---
name: fin
description: Cerrar sesión de trabajo en MSc-UTFSM. Resume lo hecho, actualiza memorias locales si aplica, escribe SESION.md como breadcrumb, y ofrece commitear los cambios. Invocar con /fin al terminar de trabajar.
---

# /fin — Cerrar sesión de trabajo en MSc-UTFSM

Cuando el usuario invoca `/fin`, ejecuta los siguientes pasos en orden.

---

## Paso 1 — Identificar qué pasó en esta sesión

Revisa el contexto de la conversación actual y responde internamente:
- ¿Qué archivos se crearon o modificaron?
- ¿Qué decisiones de diseño o arquitectura se tomaron?
- ¿Qué quedó sin terminar o bloqueado?
- ¿Cuál es el próximo paso más importante para la próxima sesión?

Si la sesión fue corta o solo exploratoria, está bien que el resumen sea breve.

---

## Paso 2 — Actualizar memorias locales (si aplica)

Evalúa si alguno de los siguientes tipos de información nueva surgió en esta sesión y merece guardarse en memoria:

| Tipo | Guardar si... |
|---|---|
| `feedback_` | El usuario corrigió tu comportamiento o confirmó un enfoque no obvio |
| `reference_` | Se leyeron papers, documentación, o se encontró un recurso externo relevante |
| `platform_` | Se tomó una decisión de arquitectura que no es obvia leyendo el código |
| `laguna_` | Se descubrió algo sobre las escenas, calibración, o resultados de Laguna Seca |

**No guardar:** cosas que están en el código, en git history, o que son obvias leyendo los archivos actuales.

Si hay algo que guardar, crea o actualiza el archivo de memoria correspondiente y agrega/actualiza la entrada en `MEMORY.md`.

---

## Paso 3 — Escribir SESION.md

Sobreescribe `sat-platform/.claude/SESION.md` con este formato exacto:

```markdown
# Última sesión — breadcrumb de trabajo

> Este archivo es actualizado por `/fin` al cerrar cada sesión.
> Viaja con git para que esté disponible en ambos computadores.
> `/inicio` lo lee para orientar el resumen de la próxima sesión.

---

## Última sesión activa
**Fecha:** [fecha de hoy]
**Computador:** [personal / universidad, según el working directory]

## Qué se hizo
[2–4 líneas concretas. Qué archivos, qué funcionalidad, qué resultados.]

## Próximo paso inmediato
[1 línea. La tarea más prioritaria para la próxima sesión.]

## Pendiente / bloqueado
[Lista breve. Omitir si no hay nada bloqueado.]
```

---

## Paso 4 — Verificar cambios locales sin commitear

Corre:
```
git status
git diff --stat
```

Lista los archivos modificados o nuevos. Si hay cambios sin commitear **que el usuario quiera guardar**, pregunta:
> ¿Commiteo los cambios antes de cerrar? (sí / no / yo lo hago)

Si responde sí, corre:
```
git add <archivos relevantes>
git commit -m "<mensaje descriptivo>"
git push origin main
```

Si SESION.md fue actualizado, siempre incluirlo en el commit.

---

## Paso 5 — Mensaje de cierre

Termina con un resumen de una sola línea:

> Sesión cerrada. Próxima sesión: [próximo paso]. Repositorio [al día / con cambios pendientes].

---

## Notas
- Si la sesión fue muy corta (solo consultas, sin cambios de código), el SESION.md puede quedarse igual — no forzar una actualización vacía.
- El commit de cierre debe tener un mensaje claro: `session: [fecha] — [qué se hizo en 5 palabras]`.
- No hacer push si el usuario dijo que prefiere hacerlo manualmente.
