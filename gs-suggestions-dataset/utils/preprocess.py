import re
from cltk.core.data_types import Doc
from cltk.alphabet.grc.grc import filter_non_greek, normalize_grc
from config.settings import tokenizer

def contains_lacunae(token: str) -> bool:
    """
    Verifica se un dato token contiene lacune (gap o parti mancanti).
    Args:
        token (str): Il token da verificare.
    Returns:
        bool: True se il token contiene lacune, False altrimenti.
    """

    if token == ".":
        return False
    if "." in token and not token.isalpha():
        return True
    if re.compile(r"(<|>|]|\[|gap|/)").search(token):
        return True
    return False


def clean_lacunae(token: str) -> str:
    """
    Pulisce il token dato rimuovendo specifici caratteri indesiderati.
    Args:
        token (str): Il token da pulire.
    Returns:
        str: Il token pulito.
    La funzione esegue i seguenti passaggi di pulizia:
    1. Se il token contiene un punto (.) e non è interamente alfabetico, lo conto come parola (token) sconosciuta.
    2. Se il token contiene uno qualsiasi dei caratteri '<', '>', ']', '[', 'gap', o '/', li rimuove.
    Esempi:
        >>> clean_lacunae("γέ.δουσιν")
        'γέδουσιν'
        >>> clean_lacunae("<")
        ''
    """

    if "." in token and not token.isalpha():
        return "<UNK>"  # Rimpiazzo con token sconosciuto
    if re.compile(r"(<|>|]|\[|gap|/|.|,|--{2,})").search(token):
        return re.sub(
            r"(<|>|]|\[|gap|/|.|,|--{2,})", "", token
        )  # Rimuovo i caratteri specificati
    return token


def greek_case_folding(text):
    """
    Esegue il case folding per il greco sul testo di input.
    Questa funzione normalizza il testo di input utilizzando la Form C di Normalizzazione Unicode (NFC)
    dopo averlo convertito in minuscolo utilizzando la Form D di Normalizzazione Unicode (NFD).
    Args:
        text (str): Il testo di input da normalizzare.
    Returns:
        str: Il testo normalizzato.
    """
    return normalize_grc(text)


def contain_gaps(text: str) -> bool:
    """
    Predicato che verifica se il testo di input contiene delle lacune (<gap/>).
    """
    return bool(re.search(r"<gap\s*/?>", text) or re.search(r"\.\.+", text))


def remove_brackets(text: str) -> str:
    """
    Rimuove le parentesi quadre dal testo di input.
    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito senza parentesi quadre.
    """
    return re.sub(r"\[(.*?)\]", r"\1", text.replace("\n", ""))


def remove_punctuation(text: str) -> str:
    """
    Rimuove la punteggiatura da una stringa di testo.
    Args:
        text (str): La stringa di testo da cui rimuovere la punteggiatura.
    Returns:
        str: La stringa di testo senza punteggiatura.
    """
    return re.sub(r"[·.,;:!?]", "", text)


def filter_dash(token_list: list[str]) -> list[str]:
    """
    Filtra una lista di token rimuovendo i trattini e concatenando i token adiacenti.
    Questa funzione scorre una lista di token e, quando trova un trattino ("-") con un token precedente e successivo,
    rimuove il trattino e concatena i due token adiacenti. Se il trattino non ha un token precedente o successivo,
    viene semplicemente ignorato.
    Args:
        token_list (list[str]): La lista di token da filtrare.
    Returns:
        list[str]: La lista di token filtrata con i trattini rimossi e i token adiacenti concatenati.
    """
    filtered_tokens = []
    i = 0
    while i < len(token_list):
        pred = succ = None
        if token_list[i] == "-" and i >= 0 and i < len(token_list):
            if i + 1 < len(token_list):
                succ = token_list[i + 1]
            if i - 1 >= 0:
                pred = token_list[i - 1]
            if pred and succ:
                concatenated_token = pred + succ
                filtered_tokens.pop()  # Rimuove il token precedente
                filtered_tokens.append(concatenated_token)
                i += 2  # Salta il token successivo
            else:
                i += 1
        else:
            filtered_tokens.append(token_list[i])
            i += 1
    return filtered_tokens


def clean_text(text: str) -> str:
    """
    Pulisce il testo di input eseguendo il case folding per il greco, la tokenizzazione e la gestione delle lacune.
    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito con i token uniti da spazi.
    """
    text = remove_brackets(remove_punctuation(text))
    tokens = tokenizer.run(input_doc=Doc(raw=text)).tokens
    cleaned_tokens = [
        clean_lacunae(token) if contains_lacunae(token) else token for token in tokens
    ]
    return filter_non_greek(" ".join(filter(None, filter_dash(cleaned_tokens))).strip())
