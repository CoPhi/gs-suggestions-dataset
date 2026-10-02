CORPUS_CHECKPOINT = "CNR-ILC/gs-dataset-train"
EVAL_CHECKPOINT = "CNR-ILC/gs-dataset-eval"
TLG_CHECKPOINT = "CNR-ILC/gs-dataset-tlg"
HERC_CHECKPOINT = "CNR-ILC/gs-dataset-herc"

BERT_UNK_TOKEN = "[UNK]"


_CORPUS_DESCRIPTION = """\
Corpus di greco antico per il pre-addestramento (MLM) di modelli BERT.

## Fonti dei dati

I testi provengono dalla combinazione delle seguenti risorse:

- **MAAT** (Machine-Actionable Ancient Text corpus) – testi papirologici ed
  epigrafici in formato EpiDoc semplificato (DDbDP, DCLP, EDH).
- **PDL** (Perseus Digital Library) – testi letterari e storiografici in greco
  antico, distribuiti con licenza aperta.
- **First1KGreek** – corpus collaborativo dei primi mille anni di letteratura
  greca (autori dal VII sec. a.C. al IV sec. d.C.).
- **TLG** (Thesaurus Linguae Graecae) – banca dati completa della letteratura
  greca antica e medievale.

## Split

| Split | Contenuto |
|-------|-----------|
| `train` | Blocchi di testo anonimizzati per MLM |
| `dev`   | Sottoinsieme di validazione (include blocchi P.Herc.) |
"""

_EVAL_DESCRIPTION = """\
Dataset di valutazione per il task di riempimento di lacune (gap filling)
su testi papirologici greci.

## Fonti dei dati

I testi di contorno provengono dalle stesse fonti del corpus di addestramento:

- **MAAT** (Machine-Actionable Ancient Text corpus)
- **PDL** (Perseus Digital Library)
- **First1KGreek**
- **TLG** (Thesaurus Linguae Graecae)

## Gold label

Le etichette di riferimento (`y`) sono le **integrazioni proposte dagli esperti
di dominio** (filologi ed editori) estratte direttamente dalle edizioni
critiche presenti nel MAAT corpus. Ogni esempio contiene:

- `x` – testo con lacuna mascherata
- `y` – lista di integrazioni accettate dagli esperti (gold label)
- `gap_length` – lunghezza stimata della lacuna
- `corpus_id` / `file_id` – identificatori del documento originale

## Split

| Split | Contenuto |
|-------|-----------|
| `dev`  | Casi di sviluppo (blocchi P.Herc.) con gold label |
| `test` | Casi di test finali con gold label |
"""

_TLG_DESCRIPTION = """\
Corpus di greco antico esclusivo del TLG (Thesaurus Linguae Graecae) per il 
pre-addestramento (MLM) di modelli BERT.
"""

_HERC_DESCRIPTION = """\
Dataset di benchmark e valutazione su testi papirologici ercolanesi (P.Herc.) per modelli linguistici (BERT e n-grammi).

## Fonti dei dati

I testi provengono dalle edizioni critiche dei Papiri di Ercolano all'interno del corpus MAAT 
(Machine-Actionable Ancient Text) e dai casi di test inediti verificati da esperti filologi (PHerc. 1004).

## Gold label

Le etichette di riferimento (`y`) sono le **integrazioni critiche proposte dagli editori papirologi**, 
estratte fedelmente dai supplementi tra parentesi quadre (`[...]`).

Ogni esempio contiene:
- `x` – testo di contesto con lacuna mascherata nel formato `[...]`
- `y` – lista delle parole/token integrati (gold labels reali)
- `gap_length` – lunghezza in caratteri alfabetici della lacuna (1–6 caratteri)
- `corpus_id` / `file_id` – identificativi del frammento e documento

## Split

| Split | Contenuto |
|-------|-----------|
| `dev`  | Casi reali estratti sistematicamente dai blocchi MAAT appartenenti a papiri di Ercolano (`P.Herc.`) |
| `test` | Casi di test controllati e inediti annotati da filologi esperti (es. PHerc. 1004) |
"""

__all__ = [
    "BERT_UNK_TOKEN",
    "CORPUS_CHECKPOINT",
    "EVAL_CHECKPOINT",
    "HERC_CHECKPOINT",
    "TLG_CHECKPOINT",
    "_CORPUS_DESCRIPTION",
    "_EVAL_DESCRIPTION",
    "_HERC_DESCRIPTION",
    "_TLG_DESCRIPTION",
]

