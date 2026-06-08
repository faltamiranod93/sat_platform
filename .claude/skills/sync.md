# /sync — Sincronizar git (alias rápido)

Para un inicio de sesión completo con resumen de contexto, usar `/inicio`.

Este skill hace solo la parte git. Cuando el usuario invoca `/sync`:

1. Corre `git pull origin main` y muestra el resultado
2. Corre `git status`
3. Corre `git log --oneline -5`
4. Informa:
   - Si llegaron cambios: qué archivos
   - Si ya estaba al día: confirmarlo
   - Si hay cambios locales sin commitear: listarlos y recordar hacer commit + push al terminar la sesión
