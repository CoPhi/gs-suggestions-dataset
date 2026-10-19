# Project Guidelines for AI Agents

## Regole di Aggiornamento della Codebase

### Aggiornamento Sistematico del `CHANGELOG.md`
Ogni volta che viene effettuata una modifica o aggiunta alla codebase:
1. **Aggiornare sempre [`CHANGELOG.md`](./CHANGELOG.md)** per riflettere i cambiamenti apportati.
2. Seguire la semantica dei **Conventional Commits** e lo standard **Keep a Changelog**:
   - `### Added` (corrispondente a `feat:`): nuove funzionalità, nuovi modelli, pipeline, script o opzioni.
   - `### Changed` (corrispondente a `refactor:` o modifiche strutturali): modifiche al comportamento o refactoring di codice esistente.
   - `### Fixed` (corrispondente a `fix:`): correzioni di bug, allineamenti di dipendenze, patch.
   - `### Optimized` / `### Performance` (corrispondente a `perf:`): ottimizzazioni di memoria, inferenza o velocità.
3. Mantenere le voci sintetiche, precise e con i riferimenti ai file/moduli modificati.
