# MAAT Light Leiden Grammar

> **Grammatica BNF del sottoinsieme "textually salient" delle Convenzioni di Leiden**  
> Estratta da: Fitzgerald & Barney (2024), *"A new machine-actionable corpus for ancient text restoration"*, ML4AL Workshop, ACL 2024.  
> Questa grammatica descrive il **formato sorgente** prodotto da MAAT dopo la conversione da EpiDoc XML: l'input del transpiler di normalizzazione per i modelli linguistici.

---

## Notazione

| Simbolo | Significato |
|---------|-------------|
| `::=` | definizione |
| `\|` | alternativa |
| `{ }` | zero o più ripetizioni |
| `[ ]` | elemento opzionale |
| `" "` | terminale letterale |
| `< >` | categoria non-terminale |
| `%` | commento |

---

## Grammatica

```bnf
<maat-text> ::= { <block-element> }
```

---

### Elementi di blocco

```bnf
<block-element> ::= <preserved-text>
                  | <restoration>
                  | <gap-known>
                  | <gap-unknown>
                  | <line-break>
                  | <whitespace>
```

---

### Testo conservato

Include le **lettere poco chiare** (*unclear*, con punto sottoscritto in Leiden standard): in MAAT sono trattate come testo ordinario. La tipografia originale (maiuscole, minuscole, accenti, spiriti, diacritici) è **preservata**.

```bnf
<preserved-text> ::= { <unicode-char> }

<unicode-char>   ::= <letter>
                   | <punctuation>
                   | <diacritic-char>

<letter>         ::= % qualsiasi lettera Unicode UTF-8
                     % (greco, latino, copto)
                     % le lettere "unclear" (ạ, ṭ, ecc.)
                     % sono normalizzate come lettere ordinarie

<diacritic-char> ::= % accenti (ὀξεῖα, βαρεῖα, περισπωμένη)
                     % spiriti (᾿ ῾)
                     % iota sottoscritto / adscrittura
                     % dieresi, altri segni Unicode

<punctuation>    ::= "." | "," | "·" | ";" | ":" | "!"
                   | "?" | "—" | "‐" | "'" | "'"
```

---

### Restauro / Supplemento editoriale

Testo integrato dal filologo. Le lacune di lunghezza **ignota** al suo interno vengono spostate **fuori** dalle parentesi da MAAT.

```bnf
<restoration>         ::= "[" <restoration-content> "]"

<restoration-content> ::= { <restored-element> }

<restored-element>    ::= <preserved-text>
                        | <gap-known>
                        % ⚠ <gap-unknown> NON può comparire dentro
                        % <restoration>: MAAT lo sposta all'esterno.
                        % Es.: Leiden [τοῦ -ca.?- παρὰ]
                        %       MAAT  [τοῦ]<gap/>[παρὰ]
```

---

### Lacuna di lunghezza nota (o approssimata)

Un punto `"."` per ogni carattere mancante. Le lacune di lunghezza **approssimata** (`ca.N` in Leiden+) sono trattate come se la lunghezza fosse nota: `N` punti consecutivi.

```bnf
<gap-known> ::= <dot> { <dot> }
<dot>       ::= "."
```

---

### Lacuna di lunghezza ignota

Corrisponde all'EpiDoc `<gap reason="lost" extent="unknown" unit="character"/>` e alle notazioni Leiden `-ca.?-`, `- - -`, `[.?]`. MAAT preserva il tag XML come stringa letterale.

```bnf
<gap-unknown> ::= "<gap/>"
                | "<gap>"
```

---

### Line break

Corrisponde all'elemento EpiDoc `<lb/>`. Preservato come carattere newline `\n`.

```bnf
<line-break> ::= "\n"
```

---

### Spazio interlineare (word space)

```bnf
<whitespace> ::= " " { " " }
```

---

## Struttura JSON del blocco anonimo (ab block)

```bnf
<ab-record> ::= "{"
                  <metadata> ","
                  <training-text-field> ","
                  <test-cases-field>
                "}"

<metadata> ::= <corpus-id-field>
             | <file-id-field>
             | <block-index-field>
             | <id-field>
             | <title-field>
             | <material-field>
             | <language-field>

<training-text-field> ::= '"training_text"' ":" '"' <maat-text> '"'

<test-cases-field>    ::= '"test_cases"' ":" "[" { <test-case> [","] } "]"

<test-case>  ::= "{"
                   '"case_index"'    ":" <integer> ","
                   '"id"'           ":" '"' <string> '"' ","
                   '"test_case"'     ":" '"' <test-case-text> '"' ","
                   '"alternatives"' ":" "[" <alternatives-list> "]"
                 "}"
```

Il **test case** è identico al training text, ma con il restauro sostituito dalla lacuna mascherata (punti `"."` pari alla **moda** delle lunghezze delle alternative).

```bnf
<test-case-text>     ::= <maat-text>

<alternatives-list>  ::= <alternative> { "," <alternative> }

<alternative>        ::= '"' <preserved-text> '"'
```

---

## Elementi Leiden scartati da MAAT

Le seguenti convenzioni **non fanno parte** del formato sorgente MAAT (vengono eliminate o ignorate nella conversione):

| Convenzione Leiden | Simbolo | Motivo esclusione |
|--------------------|---------|-------------------|
| Lettere poco chiare | `ạ`, `ṭ` (dot sottoscritto) | Normalizzate come lettere ordinarie |
| Letture alternative (apparato critico) | `<: \| :>` | Solo la prima lettura primaria viene scelta |
| Abbreviazioni non espanse | `στρατηγ(ός)` | Le parentesi tonde non vengono espanse |
| Cancellature antiche | `〚...〛` | Non preservate |
| Aggiunte soprascritte | `\testo/` | Non preservate |
| Testo espunto / surplus | `{testo}` | Non preservato |
| Correzioni di seconda mano | `$m2` | Non distinguibili nel formato MAAT |
| Vacat (spazio bianco intenzionale) | `vac.N` | Non preservato |
| Supraline, formattazione speciale | `¯...¯` | Non preservata |

---

## Vincoli semantici

I seguenti vincoli non sono esprimibili in BNF pura, ma sono parte integrante del formato:

1. La **lunghezza della maschera** in `<test-case-text>` è uguale alla **moda** delle lunghezze in `<alternatives-list>`.
2. Le alternative possono avere **lunghezze diverse** (le forme delle lettere occupano spazio diverso sul supporto fisico).
3. La codifica è **Unicode UTF-8** per tutti i campi testuali.
4. Il testo in `training_text` è lo stesso di `test_case`, ma con il restauro **integro** (gold label) al posto della lacuna.
5. `<gap-unknown>` dentro `<restoration-content>` è **ill-formed**: MAAT lo riposiziona all'esterno prima della serializzazione.

---

## Esempi (dal paper, Fig. 2)

**Training text con gap ignoto:**
```
"Ursuius vius sibi et <gap/> uxo[ri]"
```

**Test case corrispondente:**
```
"Ursuius vius sibi et <gap/> uxo[..]"
```
Alternative: `["ri"]`

---

**Training text con restauro**
```
"τὸν [πωλοῦντα]"
```

**Test case corrispondente** (7 punti = lunghezza di `πωλοῦντα`):
```
"τὸν [.......]"
```
Alternative: `["πωλοῦντα"]`

---

**Spostamento del gap fuori dal restauro:**

| | Forma |
|---|---|
| Leiden originale | `τοῦ -ca.?- [παρὰ τῆς]` |
| MAAT training text | `τοῦ <gap/> [παρὰ τῆς]` |

---

*Grammatica derivata da: Fitzgerald & Barney (2024), ML4AL Workshop, ACL 2024 — sezione 4 "Features of MAAT corpus" e sezione 5 "Format and distribution of data".*
