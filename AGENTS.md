# Project Guidelines for AI Agents

## Regole di Aggiornamento della Codebase

### Stile del Codice e Commenti Esplicativi
Quando scrivi o modifichi codice (Python, TypeScript, script o configurazioni):
1. **Commenta sempre i passaggi non autoesplicativi**: aggiungi commenti sintetici e chiari (in italiano, coerenti col resto del repository) per spiegare il *perché* di scelte algoritmiche, workaround hardware/VRAM, gestione di casi limite, retrocompatibilità o flussi reattivi non ovvi.
2. **Evita commenti ridondanti**: non commentare istruzioni banali o già autoesplicative dal nome di funzioni e variabili.

### Disaccoppiamento Architetturale e Readiness Container (`gs-api`)
Quando integri nuovi modelli o pipeline di inferenza:
1. **Esperienza Thin-Client**: l'utente finale (filologo) deve poter invocare predizioni tramite UI/API indipendentemente dall'hardware client e senza dover configurare account o token personali (es. Hugging Face).
2. **Autonomia di `gs-api`**: i pacchetti richiesti a runtime dall'API devono risiedere in `[project] dependencies` di [`pyproject.toml`](./pyproject.toml) (compatibili con `uv sync --frozen --no-dev`), mantenendo separati in `[dependency-groups] train` gli strumenti esclusivi di training/HPO.

### Aggiornamento Sistematico del `CHANGELOG.md`
Ogni volta che viene effettuata una modifica o aggiunta alla codebase:
1. **Aggiornare sempre [`CHANGELOG.md`](./CHANGELOG.md)** per riflettere i cambiamenti apportati.
2. Seguire la semantica dei **Conventional Commits** e lo standard **Keep a Changelog**:
   - `### Added` (corrispondente a `feat:`): nuove funzionalità, nuovi modelli, pipeline, script o opzioni.
   - `### Changed` (corrispondente a `refactor:` o modifiche strutturali): modifiche al comportamento o refactoring di codice esistente.
   - `### Fixed` (corrispondente a `fix:`): correzioni di bug, allineamenti di dipendenze, patch.
   - `### Optimized` / `### Performance` (corrispondente a `perf:`): ottimizzazioni di memoria, inferenza o velocità.
3. Mantenere le voci sintetiche, precise e con i riferimenti ai file/moduli modificati.

### Commit Sistematici delle Modifiche Implementate
Ogni volta che viene completata un'implementazione (codice, test, configurazioni o documentazione):
1. **Creare sempre uno o più commit** per le modifiche effettuate, senza lasciare il working tree sporco al termine del task.
2. **Raggruppare le modifiche in commit atomici** per layer architetturale (es. `frontend`, `api`, `core`, `docs`), seguendo la skill [`conventional-commits`](./.agents/skills/conventional-commits/SKILL.md).
3. **Usare i Conventional Commits** con messaggi in inglese e scope coerente col repository (es. `feat(frontend): ...`, `fix(api): ...`, `docs(agents): ...`).
4. **Includere nel commit pertinente** il codice, i test e l'aggiornamento di [`CHANGELOG.md`](./CHANGELOG.md).
