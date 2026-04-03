# Transpiler Specification: MAAT Leiden → LM-ready text

> Specifica delle trasformazioni applicate dal transpiler per convertire il formato sorgente MAAT (Leiden-like)
> in testo normalizzato pronto per l'addestramento di modelli linguistici.
>
> **Grammatica sorgente**: [maat_leiden_grammar.md](./maat_leiden_grammar.md)
> **Implementazione**: [preprocess.py](./preprocess.py), [cleaner.py](../../models/ngrams/train/cleaner.py)

---

## 1. Grammatica target (output LM-ready)

L'output del transpiler è un flusso di token normalizzati separati da spazi. Tutti i costrutti editoriali
Leiden sono stati rimossi; rimangono solo parole greche in maiuscolo e token `<UNK>` per le lacune.

```bnf
<lm-text>         ::= <lm-sentence> { <lm-sentence> }

<lm-sentence>     ::= <lm-token> { " " <lm-token> }

<lm-token>        ::= <normalized-word>
                     | <unk-token>

<normalized-word>  ::= <upper-greek-letter> { <upper-greek-letter> }
                       %% Lettere greche maiuscole, senza diacritici.
                       %% La normalizzazione Unicode è applicata tramite
                       %% CLTK normalize_grc (NFC) dopo la decomposizione.

<upper-greek-letter> ::= "Α" | "Β" | "Γ" | "Δ" | "Ε" | "Ζ" | "Η" | "Θ"
                        | "Ι" | "Κ" | "Λ" | "Μ" | "Ν" | "Ξ" | "Ο" | "Π"
                        | "Ρ" | "Σ" | "Τ" | "Υ" | "Φ" | "Χ" | "Ψ" | "Ω"
                        | "'" 
                        %% L'apostrofo è l'unico segno non alfabetico preservato 
                        %% (elisione greca, es. ΔΙ', ΑΠ')

<unk-token>        ::= "<UNK>"
                       %% Token speciale che sostituisce ogni lacuna
                       %% (sia di lunghezza nota che ignota)
```

### Proprietà della grammatica target

| Proprietà | Dettaglio |
|-----------|-----------|
| **Alfabeto** | Lettere greche maiuscole (Α–Ω), apostrofo (`'`), spazio, `<UNK>` |
| **Assenti** | `[]`, `()`, `{}`, `<>`, `†`, `‡`, `\|`, `+`, `*`, `?`, `_`, diacritici, minuscole, punteggiatura, `\n` |
| **Encoding** | Unicode UTF-8 |
| **Separatore** | Spazio singolo (`U+0020`) tra ogni token |

---

## 2. Pipeline di trasformazione

La pipeline è suddivisa in **4 fasi** sequenziali. Ogni fase consuma l'output della precedente.

```
training_text (MAAT Leiden)
      │
      ▼
 ┌─────────────────────────────────┐
 │  FASE 1: Strip markup editoriale│  process_editorial_marks()
 │  14 trasformazioni in sequenza  │  preprocess.py:651–676
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  FASE 2: Risoluzione lacune    │  clean_tokens()
 │  Per ogni token: lacuna → <UNK>│  preprocess.py:692–706
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  FASE 3: Normalizzazione greca │  normalize_greek()
 │  Rimozione diacritici + upper  │  preprocess.py:48–60
 └──────────────┬──────────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │  FASE 4: Segmentazione frasi   │  get_sentences()
 │  Sentence split + tokenizzaz.  │  cleaner.py:139–174
 │  + rimozione punteggiatura     │
 └──────────────┬──────────────────┘
                │
                ▼
        list[list[str]]
   (frasi → liste di token)
```

### Funzioni di entry point

| Entry point | Copre fasi | Contesto d'uso |
|-------------|-----------|----------------|
| `clean_text_from_gaps(text)` | 1 → 2 → 3 | Training text → stringa normalizzata |
| `get_sentences(abs)` | 1 → 2 → 3 → 4 | Blocchi anonimi → frasi tokenizzate per LM |

---

## 3. Regole di riscrittura

### Fase 1 — Strip markup editoriale (`process_editorial_marks`)

Le 14 trasformazioni sono applicate **in quest'ordine esatto**. Ogni riga è una regola `sorgente → output`.

#### T1 · Integrazioni editoriali — `process_integrations`

| Sorgente | Output | Note |
|----------|--------|------|
| `\|\|` | ` ` (spazio) | Doppie stanghe Leiden (unione di frammenti) |
| `‖` (U+2016) | ` ` (spazio) | Variante Unicode delle doppie stanghe |

#### T2 · Line break Leiden — `process_leiden_lb`

| Sorgente | Output | Note |
|----------|--------|------|
| `\|` | ` ` (spazio) | Singola stanghetta che separa le righe in Leiden |

> **Dipendenza d'ordine**: T1 deve precedere T2. Se T2 fosse applicato prima, `||` verrebbe trattato come due `|` separati producendo due spazi invece di uno.

#### T3 · Segni di incertezza — `process_unclear_signs`

| Sorgente | Output | Note |
|----------|--------|------|
| `+` | ε (rimosso) | Usato in alcune edizioni per lettere incerte |
| `*` | ε (rimosso) | Usato in alcune edizioni per lettere incerte |

#### T4 · Parentesi quadre — `process_brackets`

Questa trasformazione è composita (3 sotto-passi):

**T4a** — Pre-pulizia gap inter-frasali:
| Sorgente | Output | Regex |
|----------|--------|-------|
| `.<gap/>. ` | `.` | `\.\s*<gap/>\s*\.` → `.` |

**T4b** — Rimozione line breaks (`remove_lb`, applicato all'intero testo):
| Sorgente | Output | Regex |
|----------|--------|-------|
| `\n` seguito da cifre opzionali | ` ` (spazio) | `\n\d*` → ` ` |

**T4c** — Rimozione parentesi quadre:
| Sorgente | Output | Regex |
|----------|--------|-------|
| `[contenuto]` | `contenuto` | `\[(.*?)\]` → `\1` (non-greedy) |
| `[` o `]` isolati | ε (rimosso) | `[\[\]]` → ε |

> **Effetto collaterale critico**: dopo T4c, la distinzione tra testo conservato e testo restaurato (supplemento)
> viene **persa**. Il contenuto delle parentesi si fonde con il testo circostante.
> Per le lacune note `[..]`, i punti diventano parte del token adiacente (es. `uxo[..]` → `uxo..`),
> che verrà gestito in Fase 2.

#### T5 · Pattern dattilici — `process_dactyl_patterns`

| Sorgente | Output | Note |
|----------|--------|------|
| `⏑⏑‒`, `‒⏑⏑‒`, `‒⏑⏑`, `⏑`, `‒`, e combinazioni | `<gap/>` | Sequenze metriche in versi incompleti |

> **Produce `<gap/>`**: questa trasformazione _introduce_ nuovi token `<gap/>` nel testo. Verranno
> consumati in Fase 2 da `clean_lacunae`.

#### T6 · Vacat — `process_vacat_text`

| Sorgente | Output | Regex |
|----------|--------|-------|
| `vac.` | ε (rimosso) | `vac\.\|vacat` |
| `vacat` | ε (rimosso) | |

#### T7 · Note con doppio obelisco — `process_double_obelisks`

| Sorgente | Output | Regex |
|----------|--------|-------|
| `‡` seguito da cifre | ε (rimosso) | `‡\d+` |

#### T8 · Punti interrogativi — `process_doubts`

| Sorgente | Output | Note |
|----------|--------|------|
| `?` | ε (rimosso) | Dubbi filologici nelle edizioni |

#### T9 · Linee mancanti — `process_missing_lines`

| Sorgente | Output | Regex |
|----------|--------|-------|
| `⟦---⟧` (con uno o più `-`) | ε (rimosso) | `⟦-+\⟧` |

#### T10 · Parentesi tonde — `process_parentheses`

| Sorgente | Output | Note |
|----------|--------|------|
| `(` | ε (rimosso) | Usate per sciogliere abbreviazioni: `στρατηγ(ός)` → `στρατηγός` |
| `)` | ε (rimosso) | |

#### T11 · Aggiunte editoriali (markers `< >`) — `process_markers`

Composita (4 sotto-passi):

| Sorgente | Output | Regex / Logica |
|----------|--------|----------------|
| `&lt;contenuto&gt;` | `contenuto` | `&lt;(.*?)&gt;` → `\1` |
| `<gap/>lt;contenuto&gt;` | `<gap/>contenuto` | Recupera testo a destra di un gap con marker sinistro sconosciuto |
| `break="no"/&gt;` | ε (rimosso) | Residuo di `<lb break="no"/>` da EpiDoc |
| `&gt;`, `&lt;` isolati | ε (rimosso) | Pulizia residui |

#### T12 · Espunzioni — `process_expunctions`

| Sorgente | Output | Regex |
|----------|--------|-------|
| `{{contenuto}}` | ε (rimosso) | `\{\{(.*?)\}\}` |
| `{contenuto}` | ε (rimosso) | `\{(.*?)\}` |

#### T13 · Testo parallelo / parole morte — `process_parallel_text`

| Sorgente | Output | Regex / Logica |
|----------|--------|----------------|
| `†contenuto†` | ε (rimosso) | `†.*?†` (non-greedy) |
| `†` isolati | ε (rimosso) | |
| `_` | ε (rimosso) | Separatore in testi paralleli |

#### T14 · Trattini — `process_dash_if_needed`

Ricongiunge parole spezzate da trattini a fine riga. Applicato solo se il testo contiene `-`.

| Sorgente | Output | Esempio |
|----------|--------|---------|
| `parola-` + `fine` | `parolafine` | Ricongiungimento |
| `-fine` | (unisce a parola precedente) | `inizio` + `-fine` → `iniziofine` |
| `parola-interna` | `parolainterna` | Trattino interno rimosso |
| `-` isolato | ε (rimosso) | |

---

### Fase 2 — Risoluzione lacune (`clean_tokens`)

Dopo la Fase 1, il testo è stato liberato da tutti i costrutti editoriali.
Viene tokenizzato per spazi bianchi e ogni token viene valutato individualmente.

#### Decisione: `contains_lacunae(token)` → `bool`

```
token contiene "NONE" (case-insensitive)         → True  (placeholder)
token finisce per "." con prefisso tutto-alfa      → False (abbreviazione, non lacuna)
token contiene "<gap/>" (case-insensitive)         → True  (gap ignoto)
token contiene "." e len > 1                       → True  (punti = lacuna nota)
altrimenti                                         → False (testo normale)
```

#### Regole di riscrittura per token

| Condizione | Regola | Output | Esempio |
|------------|--------|--------|---------|
| Token senza lacuna | Conserva il token tal quale | token | `κώμης` → `κώμης` |
| Token = solo punti (`.`, `..`, `...`) | → `<UNK>` | `<UNK>` | `...` → `<UNK>` |
| Token = testo+punti (`uxo..`) | Intero token → `<UNK>` | `<UNK>` | `uxo..` → `<UNK>` ¹ |
| Token con `<gap/>` standalone | → `<UNK>` | `<UNK>` | `<gap/>` → `<UNK>` |
| Token con `<gap/>` + testo (`.λέγειν`) | Split, gap → `<UNK>`, testo conservato | `<UNK> .λέγειν` | `<gap/>.λέγειν` → `<UNK> .λέγειν` |
| Token = `None` / contiene `NONE` | → `<UNK>` | `<UNK>` | `None` → `<UNK>` |
| `<UNK>` consecutivi | Collassati in un singolo `<UNK>` | `<UNK>` | `<UNK> <UNK>` → `<UNK>` |

> ¹ **Trade-off deliberato**: quando dei punti (che indicano una lacuna di lunghezza nota) sono fusi in un token
> con lettere note (es. `uxo..` da `uxo[..]`), l'intero token diventa `<UNK>`. Le lettere note `uxo`
> vengono **sacrificate**. Questo semplifica la logica a costo di una minima perdita di informazione.

---

### Fase 3 — Normalizzazione greca (`normalize_greek`)

Applicata all'intero testo risultante dalla Fase 2.

| Passo | Operazione | Funzione | Esempio |
|-------|-----------|----------|---------|
| 3a | Decomposizione canonica (NFD) e rimozione combining characters | `strip_diacritics` | `ὅτι` → `οτι` |
| 3b | Normalizzazione Unicode CLTK | `normalize_grc` | Canonicalizzazione forme greche |
| 3c | Case folding a maiuscolo (se `case_folding=True`) | `.upper()` | `οτι` → `ΟΤΙ` |

---

### Fase 4 — Segmentazione e tokenizzazione (`get_sentences`)

Applicata in `cleaner.py` quando si preparano dati per i modelli linguistici.

| Passo | Operazione | Funzione | Input → Output |
|-------|-----------|----------|----------------|
| 4a | Tokenizzazione in frasi | `GreekRegexSentenceTokenizer.tokenize()` | `str` → `list[str]` |
| 4b | Rimozione punteggiatura (opzionale) | `remove_punctuation` | `[.·,;:!?'⁑]` → ε |
| 4c | Tokenizzazione in parole | `get_tokens_from_clean_text` (`.split()`) | `str` → `list[str]` |

---

## 4. Dipendenze d'ordine

Il seguente diagramma mostra le dipendenze obbligatorie tra le trasformazioni.
Una freccia A → B significa "A **deve** essere applicata prima di B".

```
T1 (integrations ||)
  │
  └──→ T2 (leiden lb |)       Se T2 prima di T1, || diventa due | producendo spazi extra
  
T4 (brackets [])
  │
  ├──→ Fase 2 (lacune)        I punti delle lacune note devono essere liberati dai []
  │                            prima della valutazione con contains_lacunae
  │
  └──→ T14 (dash)             I trattini dentro [] devono prima essere esposti

T5 (dactyl → <gap/>)
  │
  └──→ Fase 2 (lacune)        T5 produce <gap/> che Fase 2 converte in <UNK>

T11 (markers)
  │
  └──→ Fase 2 (lacune)        T11 può produrre <gap/> (UNKNOWN_LEFT_MARKER_REGEX)

T10 (parentheses)
  │
  └──→ T6 (vacat)             L'ordine attuale è T6 prima di T10 — indipendenti,
                               ma potenziale interazione se vacat è dentro ()

Fase 2 (lacune)
  │
  └──→ Fase 3 (normalize)     I token <UNK> non devono essere normalizzati
                               (ma .upper() su "<UNK>" è idempotente, quindi safe)

Fase 3 (normalize)
  │
  └──→ Fase 4 (sentence tok)  La sentence tokenization opera sul testo 
                               dopo la normalizzazione
```

### Trasformazioni indipendenti (ordine irrilevante tra loro)

Le seguenti trasformazioni non hanno dipendenze reciproche e potrebbero essere riordinate liberamente **tra di loro**
(ma non rispetto alle dipendenze sopra citate):

- T3 (incertezze `+`, `*`), T6 (vacat), T7 (note `‡`), T8 (dubbi `?`), T9 (linee mancanti `⟦⟧`),
  T12 (espunzioni `{}`), T13 (testo parallelo `†`)

---

## 5. Post-condizioni (invarianti dell'output)

Le seguenti proprietà devono essere sempre valide per l'output del transpiler.
Possono essere usate come base per test automatizzati di validazione.

### Dopo la Fase 1 (`process_editorial_marks`)

| ID | Invariante |
|----|-----------|
| P1.1 | Il testo non contiene `[` né `]` |
| P1.2 | Il testo non contiene `(` né `)` |
| P1.3 | Il testo non contiene `{` né `}` |
| P1.4 | Il testo non contiene `\|` (singolo) né `\|\|` (doppio) né `‖` |
| P1.5 | Il testo non contiene `+` né `*` |
| P1.6 | Il testo non contiene `?` |
| P1.7 | Il testo non contiene `†` |
| P1.8 | Il testo non contiene `vacat` né `vac.` |
| P1.9 | Il testo non contiene `⟦` né `⟧` |
| P1.10 | Il testo non contiene `\n` (rimosso da `remove_lb` in T4b) |
| P1.11 | Il testo non contiene pattern dattilici (⏑, ‒) |
| P1.12 | Il testo non contiene `-` (rimosso/ricongiunto da T14) |
| P1.13 | Il testo **può** ancora contenere `<gap/>` (prodotto da T5, T11) |
| P1.14 | Il testo **può** ancora contenere `.` (punti di lacune note liberate da T4) |
| P1.15 | Il testo **può** ancora contenere punteggiatura greca (`.`, `·`, `,`, `;`, `:`, `!`) |

### Dopo la Fase 2 (`clean_tokens`)

| ID | Invariante |
|----|-----------|
| P2.1 | Il testo non contiene `<gap/>` (tutti convertiti in `<UNK>`) |
| P2.2 | **Possono** esistere `<UNK>` consecutivi. La deduplicazione in `insert_into_clean_tokens` opera solo **intra-token** (dentro una singola invocazione di `clean_lacunae`), non tra token adiacenti. Quando più token distinti producono ciascuno un `<UNK>`, questi rimangono consecutivi. Esempio: `<gap/> ...` → `<UNK> <UNK>` |
| P2.3 | Ogni `<UNK>` è separato da spazi (risultato del `.join()`) |
| P2.4 | Nessun token contiene `.` come indicatore di lacuna (convertito in `<UNK>`) |

### Dopo la Fase 3 (`normalize_greek`)

| ID | Invariante |
|----|-----------|
| P3.1 | Il testo non contiene caratteri Unicode combining (diacritici rimossi) |
| P3.2 | Tutte le lettere sono maiuscole (se `case_folding=True`) |
| P3.3 | Il testo è in forma Unicode NFC normalizzata |

### Dopo la Fase 4 (`get_sentences`)

| ID | Invariante |
|----|-----------|
| P4.1 | Ogni frase è una lista non vuota di token |
| P4.2 | Nessun token è una stringa vuota |
| P4.3 | Il testo non contiene punteggiatura (se `remove_punct=True`) |

---

## 6. Esempi end-to-end

### Esempio 1: Supplemento pieno con gap ignoto

```
INPUT   "sibi et <gap/> uxo[ri]"
        ─────────────────────────
FASE 1  T1–T3: nessun effetto
        T4:    "sibi et <gap/> uxori"          ← [ri] → ri, fuso con uxo
        T5–T14: nessun effetto
        ─────────────────────────
FASE 2  "sibi" → sibi
        "et"   → et
        "<gap/>" → <UNK>                       ← gap ignoto
        "uxori" → uxori                        ← testo normale
        ─────────────────────────
FASE 3  "SIBI ET <UNK> UXORI"
```

### Esempio 2: Lacuna nota dentro supplemento (trade-off)

```
INPUT   "sibi et <gap/> uxo[..]"
        ─────────────────────────
FASE 1  T4:    "sibi et <gap/> uxo.."          ← [..] → .., fuso con uxo
        ─────────────────────────
FASE 2  "sibi"   → sibi
        "et"     → et
        "<gap/>" → <UNK>
        "uxo.."  → <UNK>                       ← trade-off: uxo perso
        ─────────────────────────
FASE 3  "SIBI ET <UNK> <UNK>"

        NB: due <UNK> separati perché originano da token distinti
            (uno dal gap, uno dalla lacuna). Se fossero stati
            adiacenti nello stesso token, sarebbero collassati.
```

### Esempio 3: Training text reale multi-riga

```
INPUT   "κ[ώμη]ς Σεκνεπαί\nου Νήσου ἃ ἐπεστά\nλη[σαν δ]ι' ἡμῶν"
        ─────────────────────────
FASE 1  T4b (remove_lb):
        "κ[ώμη]ς Σεκνεπαί ου Νήσου ἃ ἐπεστά λη[σαν δ]ι' ἡμῶν"
        T4c (brackets):
        "κώμης Σεκνεπαί ου Νήσου ἃ ἐπεστά λησαν δι' ἡμῶν"
        ─────────────────────────
FASE 2  Nessuna lacuna rilevata → tutti token conservati
        ─────────────────────────
FASE 3  "ΚΩΜΗΣ ΣΕΚΝΕΠΑΙ ΟΥ ΝΗΣΟΥ Α ΕΠΕΣΤΑ ΛΗΣΑΝ ΔΙ' ΗΜΩΝ"
```

### Esempio 4: Verso poetico con pattern dattilico

```
INPUT   "τὸν [πωλοῦντα] ‒⏑⏑‒"
        ─────────────────────────
FASE 1  T4:    "τὸν πωλοῦντα ‒⏑⏑‒"
        T5:    "τὸν πωλοῦντα <gap/>"            ← dattilo → gap
        ─────────────────────────
FASE 2  "τὸν"        → τὸν
        "πωλοῦντα"   → πωλοῦντα
        "<gap/>"     → <UNK>
        ─────────────────────────
FASE 3  "ΤΟΝ ΠΩΛΟΥΝΤΑ <UNK>"
```

### Esempio 5: Testo con espunzione e marker

```
INPUT   "ἄρτους &lt;τοὺς&gt; {καλούς} ε‡12"
        ─────────────────────────
FASE 1  T7:    "ἄρτους &lt;τοὺς&gt; {καλούς} ε"    ← nota rimossa
        T11:   "ἄρτους τοὺς {καλούς} ε"             ← marker → contenuto
        T12:   "ἄρτους τοὺς  ε"                      ← espunzione rimossa
        ─────────────────────────
FASE 2  Nessuna lacuna
        ─────────────────────────
FASE 3  "ΑΡΤΟΥΣ ΤΟΥΣ Ε"
```

---

*Specifica derivata dall'implementazione in [preprocess.py](./preprocess.py) e [cleaner.py](../../models/ngrams/train/cleaner.py), verificata empiricamente sulla base dati MAAT.*
