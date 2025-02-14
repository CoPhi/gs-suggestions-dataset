import re
from cltk.core.data_types import Doc
from cltk.alphabet.grc.grc import normalize_grc
from config.settings import tokenizer

LACUNAE_REGEX = re.compile(r"(<|>|]|\[|gap|/|--{2,})")
PUNCTUATION_REGEX = re.compile(r"[.·,;:!?']")
BRACKETS_REGEX = re.compile(r"\[(.*?)\]")
UNMATCHED_BRACKETS_REGEX = re.compile(r"[\[\]]")


def contains_lacunae(token: str) -> bool:
    """
    Verifica se un dato token contiene lacune (gap o parti mancanti).
    """
    return (
        token == "."
        or ("." in token and not token.isalpha())
        or bool(LACUNAE_REGEX.search(token))
    )


def clean_lacunae(token: str) -> str:
    """
    Pulisce il token dato rimuovendo specifici caratteri indesiderati.
    Args:
        token (str): Il token da pulire.
    Returns:
        str: Il token pulito.
    La funzione esegue i seguenti passaggi di pulizia:
    1. Se il token contiene un punto (.) ed è interamente alfabetico, lo conto come parola (token) sconosciuta.
    2. Se il token contiene uno qualsiasi dei caratteri '<', '>', ']', '[', 'gap', o '/', li rimuove.
    Esempi:
        >>> clean_lacunae("γέ.δουσιν")
        'UNK'
        >>> clean_lacunae("<")
        ''
    """

    if "." in token and all(char.isalpha() for char in token.replace(".", "")):
        if token.endswith(".") and all(char == "." for char in token):
            return ""
        elif token.endswith(".") and all(
            char.isalpha() for char in token.replace(".", "")
        ):
            return token
        elif all(char.isalpha() for char in token.replace(".", "")):
            return "UNK"  # Sostituisce parole con punti interni con UNK

    if LACUNAE_REGEX.search(token):  # gestione tag gap o trattini multipli
        return re.sub(LACUNAE_REGEX, "", token)  # Rimuovo i caratteri specificati

    return token


def greek_case_folding(text):
    """
    Applica il case folding greco al testo fornito.

    Args:
        text (str): Il testo in greco da normalizzare.

    Returns:
        str: Il testo normalizzato con il case folding greco applicato.
    """
    return normalize_grc(text).upper()


def remove_brackets(text: str) -> str:
    """
    Rimuove le parentesi quadre dal testo di input.
    - Se la parentesi è parte di una coppia corretta [contenuto], rimuove solo le parentesi mantenendo il contenuto.
    - Se è una parentesi aperta `[` o chiusa `]` isolata, la rimuove completamente.

    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito senza parentesi quadre.
    """
    text = BRACKETS_REGEX.sub(r"\1", text.replace("\n", " "))
    text = UNMATCHED_BRACKETS_REGEX.sub("", text)
    return text


def remove_lb(text: str) -> str:
    """
    Rimuove i caratteri di nuova riga sostituendoli con spazi.
    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito dai line breaks.

    """
    return text.replace("\n", " ")


def remove_punctuation(text: str) -> str:
    """
    Rimuove la punteggiatura da una stringa di testo.
    Args:
        text (str): La stringa di testo da cui rimuovere la punteggiatura.
    Returns:
        str: La stringa di testo senza punteggiatura.
    """
    return PUNCTUATION_REGEX.sub("", text)


def filter_dash(text: str) -> str:
    """
    Filtra i trattini presenti nel testo
    """
    return text.replace("-", "")


def get_idx_supplement_from_suppl_list(
    suppl_list: list[tuple[str, int]], supplement: str
) -> int:
    """
    Restituisce il numero di occorrenze di un supplemento nella lista suppl_list, dando un supplemento in input

    Args:
        suppl_list (list[str]): Lista di tuple (supplemento, numero di occorrenze)
        supplement (str): supplemento (stringa tra parentesi quadrate `[]`)
    """
    for idx, (suppl, num_occ) in enumerate(suppl_list):
        if suppl == supplement:
            return num_occ

    return -1


def update_num_occ_supplement_from_suppl_list(
    suppl_list: list[tuple[str, int]], supplement: str
) -> None:
    """
    Aggiorna il numero di occorrenze di un supplemento nella lista suppl_list, dando un supplemento in input

    Args:
        suppl_list (list[str]): Lista di tuple (supplemento, numero di occorrenze)
        supplement (str): supplemento (stringa tra parentesi quadrate `[]`)
    """
    for idx, (suppl, num_occ) in enumerate(suppl_list):
        if suppl == supplement:
            suppl_list[idx] = (suppl, num_occ + 1)
            return


def get_supplement_list(supplements: list[str]) -> list[tuple[str, int]]:
    """
    Restituisce una lista di tuple con i supplementi trovati nel testo e il numero di occorrenze settato a 0.

    Args:
        supplements (list[str]): Lista di supplementi.

    Returns:
        list[tuple[str, int]]: Lista di tuple (supplemento, 0).
    """
    return [(supplement, 0) for supplement in list(dict.fromkeys(supplements))]


def clean_supplements(training_text: str) -> list[list[str]]:
    """
    Questa funzione cerca i supplementi (testo racchiuso tra parentesi quadre) all'interno del testo fornito,
    estende il loro contesto se necessario, li pulisce e li tokenizza. I token vengono restituiti come
    una lista di liste, dove ogni sotto-lista rappresenta i token di un supplemento.

    Args:
        training_text (str): Il testo di addestramento contenente i supplementi da estrarre.

    Returns:
        list[list[str]]: Una lista di liste di stringhe, dove ogni lista interna contiene i token del
        corrispondente supplemento.

    Raises:
        ValueError: Se il testo fornito non è una stringa valida.
    """
    supplements = re.findall(r"(\[[^\]]+\])", training_text)  # supplementi da pulire
    suppl_list = get_supplement_list(
        supplements
    )  # lista dei supplementi a cui affianco un numero di occorrenze (da incrementare)
    suppl_tokens = []
    for suppl in supplements:
        matches = list(re.finditer(re.escape(suppl), training_text)) # Occorrenze del supplemento nel testo
        if not matches:
            suppl_tokens.append([])
            continue

        if len(matches) > 1:
            idx = get_idx_supplement_from_suppl_list(suppl_list, matches[0].group(0))
            if idx != -1:
                match = matches[idx]
                update_num_occ_supplement_from_suppl_list(
                    suppl_list, matches[0].group(0)
                )
        else:
            match = matches[0]  # Prendi la prima (unica) occorrenza trovata

        start_pos = match.start()
        end_pos = match.end()

        while start_pos > 0 and training_text[start_pos - 1] not in (" ", "\n"):
            start_pos -= 1

        while end_pos < len(training_text) and training_text[end_pos] not in (
            " ",
            "\n",
        ):
            end_pos += 1

        extended_supplement = training_text[start_pos:end_pos]

        suppl_tokens.append(
            [
                token
                for token in tokenizer.run(
                    input_doc=Doc(raw=clean_text(extended_supplement))
                ).tokens
            ]
        )

    return suppl_tokens


def clean_text_from_gaps(text: str):
    """
    Pulisce il testo dalle lacune, lasciando invariata la punteggiatura.
    Viene usato questo metodo per pulire i testi di addestramento, per poi suddividerlo in frasi.

    Args:
        text (str): testo di addestramento

    Returns:
        clean_text (str): testo pulito dalle lacune
    """
    text = filter_dash(remove_brackets(remove_lb(text)))

    tokens = tokenizer.run(input_doc=Doc(raw=text)).tokens
    cleaned_tokens = [
        clean_lacunae(token) if contains_lacunae(token) else token for token in tokens
    ]
    return " ".join(
        filter(
            None,
            [
                greek_case_folding(token) if token != "UNK" else token
                for token in cleaned_tokens
            ],
        )
    ).strip()


def clean_text(text: str) -> str:
    """
    Pulisce il testo di input eseguendo il case folding per il greco, la tokenizzazione e la gestione delle lacune.
    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito con i token uniti da spazi.
    """
    text = filter_dash(remove_brackets(remove_punctuation(remove_lb(text))))
    tokens = tokenizer.run(input_doc=Doc(raw=text)).tokens
    cleaned_tokens = [
        clean_lacunae(token) if contains_lacunae(token) else token for token in tokens
    ]
    return " ".join(
        filter(
            None,
            [
                greek_case_folding(token) if token != "UNK" else token
                for token in cleaned_tokens
            ],
        )
    ).strip()
