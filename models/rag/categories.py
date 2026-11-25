"""
Catalogo canonico delle 75 categorie letterarie TLG (<cla>) estratte da `canon.xml`.

Fornisce:
- Il catalogo statico auto-contenuto (zero dipendenze a runtime da `canon.xml` nei container).
- Normalizzazione tollerante su maiuscole, asterischi (`*Hist.`) e punti finali omessi.
- Validazione stretta e formattazione descrittiva per le Signature DSPy.
- Helper offline per estrarre `<cla>` e mappare `work_id`/`file_id` direttamente da `canon.xml`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class TLGCategory:
    """Singola categoria letteraria canonica TLG (`<cla>`)."""

    code: str
    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Serializza la categoria in dizionario JSON-ready con asse di ragionamento."""
        rc_key = get_reasoning_category_for_cla(self.code)
        rc = REASONING_CATEGORIES[rc_key]
        return {
            "code": self.code,
            "label": self.label,
            "description": self.description,
            "reasoning_category": rc_key,
            "reasoning_focus": rc["focus"],
        }


DEFAULT_TEXT_CATEGORY_PROMPT = "Non specificata (genere testuale non indicato)"

# Catalogo completo delle 75 categorie `<cla>` normalizzate presenti in `canon.xml`.
TLG_CATEGORIES: tuple[TLGCategory, ...] = (
    TLGCategory(
        code="acta",
        label="Atti / Resoconti",
        description="atti apostolici, resoconti documentari e atti civici o giudiziari",
    ),
    TLGCategory(
        code="alchem.",
        label="Alchimia",
        description="trattati alchemici, ermetici e ricettari sulla trasmutazione dei metalli",
    ),
    TLGCategory(
        code="anthol.",
        label="Antologia",
        description="raccolte antologiche ed estratti selezionati da vari autori",
    ),
    TLGCategory(
        code="apocalyp.",
        label="Apocalittica",
        description="testi apocalittici, visioni profetiche ed escatologiche",
    ),
    TLGCategory(
        code="apocryph.",
        label="Apocrifi",
        description="scritti apocrifi giudaici e cristiani, atti e vangeli apocrifi",
    ),
    TLGCategory(
        code="apol.",
        label="Apologetica",
        description="scritti apologetici e di difesa religiosa, filosofica o politica",
    ),
    TLGCategory(
        code="astrol.",
        label="Astrologia",
        description="trattati astrologici, oroscopi e dottrine sugli influssi astrali",
    ),
    TLGCategory(
        code="astron.",
        label="Astronomia",
        description="opere astronomiche, moti celesti, sfere e calendari",
    ),
    TLGCategory(
        code="biogr.",
        label="Biografia",
        description="vite di filosofi, politici, poeti e resoconti biografici",
    ),
    TLGCategory(
        code="bucol.",
        label="Poesia bucolica",
        description="idilli e composizioni pastorali in esametri, spesso in coloritura dorica",
    ),
    TLGCategory(
        code="caten.",
        label="Catene esegetiche",
        description="commenti biblici a catena compilati da più padri della Chiesa",
    ),
    TLGCategory(
        code="chronogr.",
        label="Cronografia",
        description="cronache storiche, liste di arconti/olimpiadi e computi cronologici",
    ),
    TLGCategory(
        code="comic.",
        label="Commedia",
        description="commedia attica antica, di mezzo e nuova, dialoghi scenici in trimetri giambici e metri lirici",
    ),
    TLGCategory(
        code="comm.",
        label="Commentario",
        description="commenti continui a opere filosofiche, letterarie, mediche o scientifiche",
    ),
    TLGCategory(
        code="concil.",
        label="Atti conciliari",
        description="decreti, canoni e documenti sinodali o conciliari",
    ),
    TLGCategory(
        code="coq.",
        label="Gastronomia",
        description="scritti culinari, gastronomici e ricettari antichi",
    ),
    TLGCategory(
        code="dialog.",
        label="Dialogo",
        description="dialoghi filosofici, letterari o simposiali in prosa",
    ),
    TLGCategory(
        code="doxogr.",
        label="Dossografia",
        description="raccolte di opinioni e dottrine dei filosofi antichi",
    ),
    TLGCategory(
        code="eccl.",
        label="Letteratura ecclesiastica",
        description="storia ecclesiastica, trattati disciplinari e polemica cristiana",
    ),
    TLGCategory(
        code="eleg.",
        label="Elegia",
        description="poesia elegiaca in distici elegiaci, gnomica, guerresca o amorosa, con elementi ionici",
    ),
    TLGCategory(
        code="encom.",
        label="Encomio",
        description="discorsi o componimenti encomiastici, panegirici ed epinici",
    ),
    TLGCategory(
        code="epic.",
        label="Epica",
        description="poesia epica in esametri dattilici, lessico omerico e formule epiche",
    ),
    TLGCategory(
        code="epigr.",
        label="Epigramma",
        description="epigrammi metrici brevi, votivi, funerari, simposiali o letterari",
    ),
    TLGCategory(
        code="epist.",
        label="Epistolografia",
        description="lettere reali, filosofiche, consolatorie o fittizie",
    ),
    TLGCategory(
        code="evangel.",
        label="Vangeli",
        description="testi evangelici e narrazioni neotestamentarie",
    ),
    TLGCategory(
        code="exeget.",
        label="Esegesi",
        description="commento allegorico, letterale o teologico di testi sacri o autorevoli",
    ),
    TLGCategory(
        code="fab.",
        label="Favolistica",
        description="favole esopiche e racconti morali brevi in prosa o versi",
    ),
    TLGCategory(
        code="geogr.",
        label="Geografia",
        description="trattati geografici, descrizioni di regioni, popoli e mari",
    ),
    TLGCategory(
        code="gnom.",
        label="Gnomica",
        description="sentenze, apoftegmi, massime morali e raccolte di detti celebri",
    ),
    TLGCategory(
        code="gramm.",
        label="Grammatica",
        description="trattati grammaticali, morfologia, sintassi, prosodia e ortografia",
    ),
    TLGCategory(
        code="hagiogr.",
        label="Agiografia",
        description="vite di santi, atti dei martiri e narrazioni agiografiche",
    ),
    TLGCategory(
        code="hexametr.",
        label="Poesia esametrica",
        description="componimenti didascalici o vari in esametri non strettamente eroici",
    ),
    TLGCategory(
        code="hist.",
        label="Storiografia",
        description="storie, annali e narrazioni storiografiche in prosa",
    ),
    TLGCategory(
        code="homilet.",
        label="Omiletica",
        description="omelie, sermoni e discorsi predicativi cristiani",
    ),
    TLGCategory(
        code="hymn.",
        label="Innografia",
        description="inni religiosi e cultuali in versi o prosa ritmica",
    ),
    TLGCategory(
        code="hypoth.",
        label="Hypothesis / Argomento",
        description="riassunti introduttivi e argomenti premessi a drammi o opere",
    ),
    TLGCategory(
        code="iamb.",
        label="Poesia giambica",
        description="componimenti in trimetri giambici o coliambi, satirici o didascalici",
    ),
    TLGCategory(
        code="ignotum",
        label="Genere ignoto",
        description="opere frammentarie di genere non identificabile nel canone TLG",
    ),
    TLGCategory(
        code="invectiv.",
        label="Invettiva",
        description="scritti polemici, biasimi e invettive retoriche o letterarie",
    ),
    TLGCategory(
        code="jurisprud.",
        label="Giurisprudenza",
        description="testi giuridici, leggi, costituzioni imperiali e trattati di diritto",
    ),
    TLGCategory(
        code="lexicogr.",
        label="Lessicografia",
        description="lessici, glossari, etimologici e raccolte di voci rare o dialettali",
    ),
    TLGCategory(
        code="liturg.",
        label="Liturgia",
        description="preghiere, formule liturgiche, inni e testi rituali",
    ),
    TLGCategory(
        code="lyr.",
        label="Poesia lirica",
        description="lirica monodica e corale, odi e frammenti melici, spesso in dialetti eolico/dorico",
    ),
    TLGCategory(
        code="magica",
        label="Testi magici",
        description="papiri magici, incantesimi, invocazioni e rituali teurgici",
    ),
    TLGCategory(
        code="math.",
        label="Matematica",
        description="trattati di aritmetica, geometria e matematica teorica",
    ),
    TLGCategory(
        code="mech.",
        label="Meccanica",
        description="trattati di meccanica, ingegneria, macchine e pneumatica",
    ),
    TLGCategory(
        code="med.",
        label="Medicina",
        description="trattati medici, patologia, anatomia, farmacologia e dietetica",
    ),
    TLGCategory(
        code="metrolog.",
        label="Metrologia",
        description="scritti su pesi, misure, monete e sistemi metrologici",
    ),
    TLGCategory(
        code="mim.",
        label="Mimo",
        description="mimi e scenette realistiche dialogate in prosa o in versi",
    ),
    TLGCategory(
        code="mus.",
        label="Musica",
        description="trattati di teoria musicale, armonica e ritmica",
    ),
    TLGCategory(
        code="myth.",
        label="Mitografia",
        description="raccolte mitografiche, genealogie eroiche e narrazioni di miti",
    ),
    TLGCategory(
        code="narr.fict.",
        label="Narrativa di finzione",
        description="romanzi greci, novelle e narrazioni romanzesche",
    ),
    TLGCategory(
        code="nat.hist.",
        label="Storia naturale",
        description="scienze naturali, zoologia, botanica, fisica e mineralogia",
    ),
    TLGCategory(
        code="onir.",
        label="Onirocritica",
        description="interpretazione dei sogni e trattati onirocritici",
    ),
    TLGCategory(
        code="orac.",
        label="Oracoli",
        description="responsi oracolari, raccolte sibilline e profezie in esametri o prosa",
    ),
    TLGCategory(
        code="orat.",
        label="Oratoria",
        description="discorsi giudiziari, deliberativi ed epidittici degli oratori attici e imperiali",
    ),
    TLGCategory(
        code="paradox.",
        label="Paradossografia",
        description="raccolte di fenomeni meravigliosi, curiosità naturali e paradossi",
    ),
    TLGCategory(
        code="parod.",
        label="Parodia",
        description="parodie epiche o letterarie e testi burleschi",
    ),
    TLGCategory(
        code="paroem.",
        label="Paremiografia",
        description="raccolte e spiegazioni di proverbi e modi di dire antichi",
    ),
    TLGCategory(
        code="perieg.",
        label="Periegesi / Periplo",
        description="descrizioni di viaggi, peripli marittimi e itinerari",
    ),
    TLGCategory(
        code="phil.",
        label="Filosofia",
        description="trattati filosofici, etica, logica, fisica, metafisica, inclusi testi epicurei e stoici",
    ),
    TLGCategory(
        code="physiognom.",
        label="Fisiognomica",
        description="trattati sull'interpretazione del carattere dai tratti fisici",
    ),
    TLGCategory(
        code="poem.",
        label="Poesia varia",
        description="componimenti poetici di metro o genere vario non ascritti a una sola forma",
    ),
    TLGCategory(
        code="polyhist.",
        label="Erudizione / Poligrafia",
        description="miscellanee erudite, questioni conviviali ed enciclopedismo antico",
    ),
    TLGCategory(
        code="prophet.",
        label="Letteratura profetica",
        description="libri profetici biblici e oracoli profetici",
    ),
    TLGCategory(
        code="pseudepigr.",
        label="Pseudoepigrafi",
        description="scritti pseudoepigrafi giudaico-ellenistici e apocrifi veterotestamentari",
    ),
    TLGCategory(
        code="relig.",
        label="Letteratura religiosa",
        description="scritti religiosi, cultuali e devozionali",
    ),
    TLGCategory(
        code="rhet.",
        label="Retorica",
        description="trattati di teoria retorica, stile, figure, declamazioni e critica letteraria",
    ),
    TLGCategory(
        code="satura",
        label="Satira",
        description="scritti satirici, menippei e dialoghi ironici",
    ),
    TLGCategory(
        code="satyr.",
        label="Dramma satiresco",
        description="drammi satireschi in versi di registro misto tragico-comico",
    ),
    TLGCategory(
        code="schol.",
        label="Scoli",
        description="glosse e note marginali esplicative tramandate nei manoscritti di autori classici",
    ),
    TLGCategory(
        code="tact.",
        label="Tattica / Arte militare",
        description="trattati di strategia, poliorcetica e arte militare",
    ),
    TLGCategory(
        code="test.",
        label="Testimonia",
        description="testimonianze antiche indirette sulla vita e le opere di un autore",
    ),
    TLGCategory(
        code="theol.",
        label="Teologia",
        description="trattati teologici, dogmatici e speculazione sul divino",
    ),
    TLGCategory(
        code="trag.",
        label="Tragedia",
        description="tragedia attica ed ellenistica, registro elevato, trimetri giambici e cori dorizzanti",
    ),
)

TLG_CATEGORY_BY_CODE: dict[str, TLGCategory] = {cat.code: cat for cat in TLG_CATEGORIES}

# Categorie di ragionamento filologico che raggruppano i 75 codici `<cla>` in base
# agli assi interpretativi utilizzati nell'ottimizzazione zero-shot delle istruzioni DSPy.
REASONING_CATEGORIES: dict[str, dict[str, Any]] = {
    "poesia_metrica": {
        "label": "Poesia e metrica",
        "focus": "vincoli metrici/prosodici, coloritura dialettale poetica (omerico/ionico/dorico/eolico) e lessico formulare",
        "cues": (
            "metr",
            "esametr",
            "verso",
            "poet",
            "dialett",
            "omeric",
            "doric",
            "ionic",
            "eolic",
            "epitet",
            "formular",
            "prosod",
            "giamb",
            "eleg",
            "liric",
            "tragic",
            "comic",
            "coro",
            "strof",
            "epigramm",
        ),
        "codes": (
            "bucol.",
            "comic.",
            "eleg.",
            "epic.",
            "epigr.",
            "hexametr.",
            "hymn.",
            "iamb.",
            "lyr.",
            "orac.",
            "parod.",
            "poem.",
            "satyr.",
            "trag.",
        ),
    },
    "filosofia_argomentazione": {
        "label": "Filosofia e argomentazione",
        "focus": "terminologia tecnica filosofica, nessi logico-argomentativi e coerenza dottrinale (es. epicurea, stoica, peripatetica)",
        "cues": (
            "filosof",
            "dottrin",
            "argoment",
            "logic",
            "epicure",
            "stoic",
            "etic",
            "fisic",
            "concett",
            "ragion",
            "dimostraz",
            "teori",
            "trattat",
            "dialog",
            "dossograf",
            "comment",
        ),
        "codes": ("comm.", "dialog.", "doxogr.", "phil."),
    },
    "retorica_prosa": {
        "label": "Retorica, oratoria e prosa letteraria",
        "focus": "sintassi periodale, figure retoriche, registro prosastico/narrativo e usi oratori o storiografici",
        "cues": (
            "retoric",
            "orator",
            "pros",
            "sintass",
            "period",
            "stil",
            "registr",
            "narrat",
            "storiograf",
            "discors",
            "encomi",
            "invettiv",
            "epistol",
            "biograf",
            "satir",
            "favol",
            "geograf",
            "mim",
        ),
        "codes": (
            "biogr.",
            "chronogr.",
            "encom.",
            "epist.",
            "fab.",
            "geogr.",
            "hist.",
            "invectiv.",
            "mim.",
            "narr.fict.",
            "orat.",
            "paradox.",
            "perieg.",
            "polyhist.",
            "rhet.",
            "satura",
        ),
    },
    "scienze_tecnica": {
        "label": "Scienze, medicina e letteratura tecnica",
        "focus": "lessico tecnico-scientifico specialistico, precisione terminologica e formule descrittive o prescrittive",
        "cues": (
            "medic",
            "tecnic",
            "scientific",
            "matemat",
            "astronom",
            "astrolog",
            "meccanic",
            "music",
            "natural",
            "anatom",
            "terap",
            "ricett",
            "misur",
            "tattic",
            "alchim",
            "fisiognom",
            "sogn",
            "onir",
        ),
        "codes": (
            "alchem.",
            "astrol.",
            "astron.",
            "coq.",
            "math.",
            "mech.",
            "med.",
            "metrolog.",
            "mus.",
            "nat.hist.",
            "onir.",
            "physiognom.",
            "tact.",
        ),
    },
    "grammatica_esegesi": {
        "label": "Grammatica, lessicografia ed esegesi",
        "focus": "analisi morfologica e grammaticale, glosse, citazioni intertestuali e apparato scoliastico/gnomico",
        "cues": (
            "grammat",
            "lessic",
            "morfolog",
            "gloss",
            "scoli",
            "esege",
            "citazion",
            "proverb",
            "sentenz",
            "gnom",
            "antolog",
            "mitograf",
            "argoment",
            "testimoni",
            "etimo",
        ),
        "codes": (
            "anthol.",
            "gnom.",
            "gramm.",
            "hypoth.",
            "lexicogr.",
            "myth.",
            "paroem.",
            "schol.",
            "test.",
        ),
    },
    "religione_diritto": {
        "label": "Testi religiosi, teologici, giuridici e documentari",
        "focus": "lessico teologico, biblico, patristico, liturgico o giuridico e formulari rituali/documentari",
        "cues": (
            "teolog",
            "religi",
            "biblic",
            "patristic",
            "eccles",
            "omeli",
            "liturg",
            "evangel",
            "apocali",
            "apocrif",
            "apolog",
            "agiograf",
            "concil",
            "giuridic",
            "legg",
            "magic",
            "cult",
            "profet",
            "caten",
            "att",
        ),
        "codes": (
            "acta",
            "apocalyp.",
            "apocryph.",
            "apol.",
            "caten.",
            "concil.",
            "eccl.",
            "evangel.",
            "exeget.",
            "hagiogr.",
            "homilet.",
            "jurisprud.",
            "liturg.",
            "magica",
            "prophet.",
            "pseudepigr.",
            "relig.",
            "theol.",
        ),
    },
    "neutro": {
        "label": "Analisi filologica generale",
        "focus": "morfologia, sintassi, coerenza contestuale e riscontro sui loci similes",
        "cues": (
            "morfolog",
            "sintass",
            "contest",
            "attestat",
            "parallel",
            "lacun",
            "letter",
            "parol",
            "radic",
            "prefiss",
            "suffiss",
            "lessic",
        ),
        "codes": ("ignotum",),
    },
}

_CLA_TO_REASONING_CATEGORY: dict[str, str] = {
    code: rc_key
    for rc_key, rc_meta in REASONING_CATEGORIES.items()
    for code in rc_meta["codes"]
}
assert set(_CLA_TO_REASONING_CATEGORY.keys()) == set(TLG_CATEGORY_BY_CODE.keys()), (
    "Copertura incompleta delle 75 categorie <cla> in REASONING_CATEGORIES"
)


def get_reasoning_category_for_cla(code: str) -> str:
    """Restituisce l'identificativo dell'asse di ragionamento filologico associato al `<cla>`."""
    canonical = normalize_text_category(code)
    return _CLA_TO_REASONING_CATEGORY.get(canonical, "neutro")


def get_reasoning_profile(categories: str | Sequence[str] | None) -> dict[str, Any]:
    """
    Restituisce il profilo di ragionamento aggregato (chiavi, etichette, focus e cue lessicali)
    per una o più categorie `<cla>`, ricadendo su `'neutro'` se non specificate.
    """
    codes = normalize_text_categories(categories)
    if not codes:
        rc = REASONING_CATEGORIES["neutro"]
        return {
            "reasoning_categories": ["neutro"],
            "labels": [rc["label"]],
            "focuses": [rc["focus"]],
            "cues": list(rc["cues"]),
        }

    rc_keys: list[str] = []
    seen_rc: set[str] = set()
    for code in codes:
        rk = get_reasoning_category_for_cla(code)
        if rk not in seen_rc:
            seen_rc.add(rk)
            rc_keys.append(rk)

    labels: list[str] = []
    focuses: list[str] = []
    cues: list[str] = []
    seen_cues: set[str] = set()
    for rk in rc_keys:
        rc = REASONING_CATEGORIES[rk]
        labels.append(rc["label"])
        focuses.append(rc["focus"])
        for cue in rc["cues"]:
            if cue not in seen_cues:
                seen_cues.add(cue)
                cues.append(cue)

    return {
        "reasoning_categories": rc_keys,
        "labels": labels,
        "focuses": focuses,
        "cues": cues,
    }

# Mappa tollerante priva di punti/spazi per accettare varianti senza punteggiatura (es. 'phil' -> 'phil.')
_ALIAS_TO_CANONICAL_CODE: dict[str, str] = {
    re.sub(r"[\s.]+", "", cat.code.lower()): cat.code
    for cat in TLG_CATEGORIES
}

_NEUTRAL_CATEGORY_STRINGS = frozenset(
    {
        "",
        "none",
        "null",
        "non specificata",
        DEFAULT_TEXT_CATEGORY_PROMPT.lower(),
    }
)


def normalize_raw_cla_tag(raw: str) -> str:
    """Normalizza un tag `<cla>` grezzo estratto da `canon.xml` (asterischi, maiuscole, punti omessi)."""
    cleaned = re.sub(r"\s+", "", (raw or "").strip().lstrip("*").lower())
    if cleaned == "narr.fict":
        return "narr.fict."
    if cleaned == "nat.hist":
        return "nat.hist."
    return cleaned


def normalize_text_category(category: str) -> str:
    """
    Normalizza un singolo codice `<cla>` in forma tollerante (maiuscole/minuscole,
    asterisco iniziale `*`, punti o spazi interni omessi) e valida che appartenga
    al catalogo canonico delle 75 categorie TLG.
    """
    if not isinstance(category, str):
        raise ValueError(
            f"Categoria TLG non riconosciuta: {category!r}. "
            "Usa uno dei 75 codici <cla> canonici (es. 'phil.', 'epic.', 'rhet.', 'hist.', 'comic.')."
        )
    key = re.sub(r"[\s.]+", "", category.strip().lstrip("*").lower())
    canonical = _ALIAS_TO_CANONICAL_CODE.get(key)
    if canonical is None:
        raise ValueError(
            f"Categoria TLG non riconosciuta: {category!r}. "
            "Usa uno dei 75 codici <cla> canonici (es. 'phil.', 'epic.', 'rhet.', 'hist.', 'comic.')."
        )
    return canonical


def normalize_text_categories(categories: str | Sequence[str] | None) -> list[str]:
    """
    Normalizza 0, 1 o più categorie `<cla>` in una lista deduplicata di codici canonici.
    Accetta `None`, stringa singola/separata da virgole o `;`, oppure una sequenza di stringhe.
    """
    if categories is None:
        return []

    raw_tokens: list[str] = []
    if isinstance(categories, str):
        stripped = categories.strip()
        if stripped.lower() in _NEUTRAL_CATEGORY_STRINGS:
            return []
        raw_tokens.extend(tok.strip() for tok in re.split(r"[,;]+", stripped) if tok.strip())
    else:
        for item in categories:
            if item is None:
                continue
            item_str = str(item).strip()
            if not item_str or item_str.lower() in _NEUTRAL_CATEGORY_STRINGS:
                continue
            raw_tokens.extend(tok.strip() for tok in re.split(r"[,;]+", item_str) if tok.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        code = normalize_text_category(tok)
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return deduped


def format_text_category_for_prompt(categories: str | Sequence[str] | None) -> str:
    """
    Formatta una o più categorie `<cla>` in una descrizione leggibile per l'LLM nel prompt DSPy,
    restituendo `DEFAULT_TEXT_CATEGORY_PROMPT` quando nessuna categoria è specificata.
    """
    if isinstance(categories, str) and categories.strip() == DEFAULT_TEXT_CATEGORY_PROMPT:
        return DEFAULT_TEXT_CATEGORY_PROMPT

    codes = normalize_text_categories(categories)
    if not codes:
        return DEFAULT_TEXT_CATEGORY_PROMPT

    parts = []
    for code in codes:
        cat = TLG_CATEGORY_BY_CODE[code]
        rc = REASONING_CATEGORIES[get_reasoning_category_for_cla(code)]
        parts.append(
            f"{cat.label} ({code}) — {cat.description} "
            f"[Categoria di ragionamento: {rc['label']}; focus: {rc['focus']}]"
        )
    return "; ".join(parts)


def list_tlg_categories() -> list[dict[str, str]]:
    """Restituisce l'intero catalogo delle 75 categorie `<cla>` come lista di dizionari."""
    return [cat.to_dict() for cat in TLG_CATEGORIES]


def extract_cla_categories_from_canon(canon_xml_path: str | Path) -> list[str]:
    """
    Estrae e restituisce i codici `<cla>` unici e normalizzati presenti nei tag `<work>`
    del file XML `canon.xml`.
    """
    path = Path(canon_xml_path)
    if not path.exists():
        raise FileNotFoundError(f"File canon.xml non trovato: '{path}'.")

    tree = ET.parse(path)
    root = tree.getroot()
    found: set[str] = set()
    for work in root.iter("work"):
        for cla in work.findall("cla"):
            raw = (cla.attrib.get("value") or cla.text or "").strip()
            if not raw:
                continue
            norm = normalize_raw_cla_tag(raw)
            if norm:
                found.add(normalize_text_category(norm))
    return sorted(found)


def load_tlg_work_categories_from_canon(canon_xml_path: str | Path) -> dict[str, list[str]]:
    """
    Costruisce una mappa `{chiave_opera: [codici_cla_normalizzati]}` da `canon.xml`.
    Popola sia il `file_id` (`tlgXXXXYYY.xml`) sia il `work_id` (`XXXX YYY`).
    """
    path = Path(canon_xml_path)
    if not path.exists():
        return {}

    tree = ET.parse(path)
    root = tree.getroot()
    work_map: dict[str, list[str]] = {}

    for work in root.iter("work"):
        wid = (work.attrib.get("id") or "").strip()
        if not wid:
            continue
        cats: list[str] = []
        seen: set[str] = set()
        for cla in work.findall("cla"):
            raw = (cla.attrib.get("value") or cla.text or "").strip()
            if not raw:
                continue
            try:
                code = normalize_text_category(normalize_raw_cla_tag(raw))
            except ValueError:
                continue
            if code not in seen:
                seen.add(code)
                cats.append(code)
        if not cats:
            continue

        work_map[wid] = cats
        parts = wid.split()
        if len(parts) == 2:
            work_map[f"tlg{parts[0]}{parts[1]}.xml"] = cats

    return work_map
