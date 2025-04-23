import re
from cltk.alphabet.grc.grc import normalize_grc
from utils import (
    GAP_TOKEN,
    UNK_TOKEN,
    PUNCTUATION_REGEX,
    BRACKETS_REGEX,
    UNMATCHED_BRACKETS_REGEX,
    SUPPLEMENTS_REGEX,
    MISSING_LINES_REGEX,
    EXPUNCTION_REGEX,
    MARKER_REGEX,
    UNKNOWN_LEFT_MARKER_REGEX,
    EXTENDED_LINE_RIGHT_MARKER_REGEX,
    VACAT_REGEX,
    NOTES_REGEX,
    OBELISK_REGEX,
    COMBINED_DACTYL_PATTERNS,
)


def contains_lacunae(token: str) -> bool:
    """
    Verifica se un dato token contiene lacune da processare.
    """
    fold_token = token.upper()

    if "NONE" in fold_token:
        return True

    if token.endswith(".") and all(char.isalpha() for char in token[:-1]):
        return False

    return (GAP_TOKEN.upper() in fold_token) or ("." in token and len(token) > 1)


def normalize_greek(text: str) -> str:
    """
    Applica il case folding greco al testo fornito.

    Args:
        text (str): Il testo in greco da normalizzare.

    Returns:
        str: Il testo normalizzato con il case folding greco applicato.
    """
    return normalize_grc(text)


def clean_lacunae(token: str) -> str:
    """
    Pulisce il token dato rimuovendo specifici caratteri indesiderati.

    Args:
        token (str): Il token da pulire.

    Returns:
        str: Il token pulito.

    Se il token contiene uno qualsiasi dei caratteri che contraddistinguono una lacuna, come '<gap/>' o dei semplici punti, si ritorna un token sconosciuto `<UNK>`.

    Esempi:
        >>> clean_lacunae("γέ.δουσιν")
        '<UNK>'
        >>> clean_lacunae("....")
        '<UNK>'
        >>> clean_lacunae("<gap/>.λέγειν")
        '<UNK> .λέγειν'
    """
    if GAP_TOKEN in token:  # gestione tag gap
        clean_seq = []
        seq = token.replace(GAP_TOKEN, " <UNK> ").split()
        for i, tkn in enumerate(seq):
            if (
                tkn.endswith(".")
                and all(char.isalpha() or char.isspace() for char in tkn[:-1])
                and "NONE" not in tkn.upper()
            ) and (clean_seq and clean_seq[-1] == UNK_TOKEN):
                clean_seq = insert_into_clean_tokens(clean_seq, UNK_TOKEN)

            elif is_part_of_lacuna(tkn):
                clean_seq = insert_into_clean_tokens(clean_seq, UNK_TOKEN)
            else:
                if i == len(seq) - 1:
                    clean_seq = insert_into_clean_tokens(clean_seq, tkn)
                else:
                    if (
                        tkn.endswith(".")
                        and all(char.isalpha() or char.isspace() for char in tkn[:-1])
                        and "NONE" not in tkn.upper()
                    ):
                        clean_seq = clean_seq + [tkn]
                    else:
                        next_seq = clean_lacunae(" ".join(seq[i:])).replace(
                            " <UNK> ", GAP_TOKEN
                        )
                        clean_seq = clean_seq + next_seq.split()
        return " ".join(clean_seq).strip()

    return UNK_TOKEN  # Caso in cui contiene punti o None (lacune di lunghezza approssimata o definita)


def is_part_of_lacuna(token: str) -> bool:
    """
    Verifica se il token dato è parte di una lacuna.

    Args:
        token (str): Il token da verificare.

    Returns:
        bool: True se il token è parte di una lacuna, False altrimenti.
    """

    if token == ".":
        return True

    if (
        token.startswith(".")
        and all(char.isalpha() or char.isspace() for char in token[1:])
        and "NONE" not in token.upper()
    ):
        return False

    return bool(
        all(char.isalpha() for char in token)
        or contains_lacunae(token)
        or token == UNK_TOKEN
    )


def insert_into_clean_tokens(clean_list: list[str], token: str) -> list[str]:
    """
    Inserisce un token nella lista dei token puliti, se il token non è vuoto.

    Args:
        clean_list (list[str]): La lista di token puliti.
        token (str): Il token da inserire.
    """

    if not token:
        return clean_list

    if not clean_list and token:
        clean_list.append(token)
        return clean_list

    if token == UNK_TOKEN and clean_list[-1] == UNK_TOKEN:
        return clean_list

    clean_list.append(token)
    return clean_list


def remove_brackets(text: str) -> str:
    """
    Rimuove le parentesi quadre mantenendo il contenuto interno.
    Se le parentesi sono isolate, le rimuove completamente.

    Args:
        text (str): Il testo da processare.

    Returns:
        str: Testo senza parentesi quadre.
    """
    text = BRACKETS_REGEX.sub(r"\1", remove_lb(text))
    text = UNMATCHED_BRACKETS_REGEX.sub("", text)
    return text.strip()


def remove_lb(text: str) -> str:
    """
    Rimuove i caratteri di nuova riga (Line Breaks `\\n`) e numeri di linea da un testo sostituendoli con spazi.
    Args:
        text (str): Il testo di input da pulire.
    Returns:
        str: Il testo pulito dai line breaks e numeri di linea.

    """
    return re.sub(r"\n\d*", " ", text)


def remove_punctuation(text: str) -> str:
    """
    Rimuove la punteggiatura da una stringa di testo.
    Args:
        text (str): La stringa di testo da cui rimuovere la punteggiatura.
    Returns:
        str: La stringa di testo senza punteggiatura.
    """
    return PUNCTUATION_REGEX.sub("", text)


def process_dash(words: list[str], result: list[str], i: int) -> tuple[list[str], int]:
    """
    Processa i trattini presenti nelle parole e li rimpiazza con le parole circostanti.

    Args:
        words (list[str]): Lista di parole da processare.
        result (list[str]): Lista di parole processate.
        i (int): Indice corrente nella lista delle parole.

    Returns:
        tuple[list[str], int]: Lista di parole processate e nuovo indice.
    """
    if words[i] == "-":
        return result, i + 1
    elif words[i].endswith("-"):
        return process_ending_dash(words, result, i)
    elif words[i].startswith("-"):
        return process_starting_dash(words, result, i)
    else:
        w = words[i].replace("-", "")
        if w:
            result.append(w)
        return result, i + 1


def process_ending_dash(
    words: list[str], result: list[str], i: int
) -> tuple[list[str], int]:
    """
    Processa i trattini presenti alla fine delle parole.

    Args:
        words (list[str]): Lista di parole da processare.
        result (list[str]): Lista di parole processate.
        i (int): Indice corrente nella lista delle parole.

    Returns:
        tuple[list[str], int]: Lista di parole processate e nuovo indice.
    """
    if i + 1 < len(words):
        next_word = words[i + 1]
        if "-" in next_word:
            return result, i + 2
        result.append(words[i].replace("-", "") + next_word)
        return result, i + 2
    else:
        result.append(words[i].replace("-", ""))
        return result, i + 1


def process_starting_dash(
    words: list[str], result: list[str], i: int
) -> tuple[list[str], int]:
    """
    Processa i trattini presenti all'inizio delle parole.

    Args:
        words (list[str]): Lista di parole da processare.
        result (list[str]): Lista di parole processate.
        i (int): Indice corrente nella lista delle parole.

    Returns:
        tuple[list[str], int]: Lista di parole processate e nuovo indice.
    """
    if i - 1 >= 0:
        prev_word = result.pop() if result else ""
        result.append(prev_word + words[i].replace("-", ""))
        return result, i + 1
    else:
        result.append(words[i].replace("-", ""))
        return result, i + 1


def filter_dash(text: str) -> str:
    """
    Filtra i trattini presenti nel testo rimpiazzandoli con le parole circostanti.

    Args:
        text (str): Il testo di input contenente i trattini.

    Returns:
        str: Il testo con i trattini rimpiazzati dalle parole circostanti.
    """
    words = get_tokens_from_clean_text(text)
    result = []
    i = 0

    while i < len(words):
        if "-" in words[i]:
            result, i = process_dash(words, result, i)
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


def get_num_occ_supplement_from_suppl_dict(suppl_dict: dict, supplement: str) -> int:
    """
    Restituisce il numero di occorrenze di un supplemento nella lista suppl_dict, dando un supplemento in input

    Args:
        suppl_dict (dict): Dizionario { supplemento : numero di occorrenze }
        supplement (str): supplemento (stringa tra parentesi quadrate `[]`)
    """
    return suppl_dict.get(supplement, -1)


def update_num_occ_supplement_from_suppl_dict(
    suppl_dict: dict, supplement: str
) -> None:
    """
    Aggiorna il numero di occorrenze di un supplemento nella lista suppl_dict, dando un supplemento in input

    Args:
        suppl_dict (dict): Dizionario { supplemento : numero di occorrenze }
        supplement (str): supplemento (sottostringa presente nel testo di addestramento tra parentesi quadrate `[supplemento]`)
    """
    if supplement in suppl_dict:
        suppl_dict[supplement] += 1


def get_supplement_dict(supplements: list[str]) -> dict:
    """
    Restituisce un dizionario con i supplementi trovati nel testo e il numero di occorrenze settato a 0.

    Args:
        supplements (list[str]): Lista di supplementi.

    Returns:
        dict: Lista di tuple (supplemento, 0).
    """
    return {supplement: 0 for supplement in supplements}


def suppl_contain_end_space(training_text: str, end_pos: int):
    """
    Verifica se il supplemento contiene uno spazio alla fine
    """
    return training_text[end_pos - 2] == " "


def suppl_contain_start_space(training_text: str, start_pos: int):
    """
    Verifica se il supplemento contiene uno spazio all'inizio
    """
    return training_text[start_pos + 1] == " "


def get_extended_supplement(training_text: str, start_pos: int, end_pos: int):
    """
    Estrae il supplemento dal testo di addestramento, estendendolo se necessario.
    """

    if suppl_contain_start_space(training_text, start_pos) and suppl_contain_end_space(
        training_text, end_pos
    ):
        return training_text[start_pos + 1 : end_pos - 2]

    if suppl_contain_end_space(training_text, end_pos):
        # Mi sposto solo a sinistra
        while start_pos > 0 and training_text[start_pos - 1] not in (" ", "\n"):
            start_pos -= 1
        return training_text[start_pos : end_pos - 1]

    if suppl_contain_start_space(training_text, start_pos):
        # Mi sposto solo a destra
        while end_pos < len(training_text) and training_text[end_pos] not in (
            " ",
            "\n",
        ):
            end_pos += 1
        return training_text[start_pos:end_pos]

    while start_pos > 0 and training_text[start_pos - 1] not in (" ", "\n"):
        start_pos -= 1

    while end_pos < len(training_text) and training_text[end_pos] not in (
        " ",
        "\n",
    ):
        end_pos += 1

    return training_text[start_pos:end_pos]


def clean_supplements(training_text: str, case_folding: bool = True) -> list[list[str]]:
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
    supplements = SUPPLEMENTS_REGEX.findall(training_text)  # supplementi da pulire
    suppl_dict = get_supplement_dict(
        supplements
    )  # dizionario dei supplementi a cui affianco un numero di occorrenze (da incrementare)
    suppl_tokens = []

    for suppl in supplements:
        matches = list(
            re.finditer(re.escape(suppl), training_text) #Si usa re.escape per trattare il supplemento come stringa
        )  # Occorrenze del supplemento nel testo
        if not matches:
            suppl_tokens.append([])
            continue

        if len(matches) > 1:
            idx = get_num_occ_supplement_from_suppl_dict(suppl_dict, matches[0].group(0))
            if idx != -1:
                match = matches[idx]
                update_num_occ_supplement_from_suppl_dict(
                    suppl_dict, matches[0].group(0)
                )
        else:
            match = matches[0]  # Prendi la prima (unica) occorrenza trovata

        extended_supplement = get_extended_supplement(
            training_text, match.start(), match.end()
        )

        suppl_tokens.append(
            [
                token
                for token in get_tokens_from_clean_text(
                    remove_punctuation(
                        clean_text_from_gaps(extended_supplement, case_folding)
                    )
                )
            ]
        )
    return suppl_tokens


def get_tokens_from_clean_text(text: str) -> list[str]:
    """
    Estrae i token da un testo pulito tramite il metodo `clean_text_from_gaps()`.

    Args:
        text (str): Il testo pulito da cui estrarre i token.

    Returns:
        list[str]: Una lista di token estratti dal testo pulito.
    """
    return text.split()


def process_token(token: str) -> list[str]:
    """
    Funzione usata all'interno di `clean_text_from_gaps` per restituire i token puliti.
    Pulisce o normalizza un token greco se non sono presenti lacune.
    Questa funzione verifica se il token contiene lacune e, in tal caso, applica la funzione `clean_lacunae`.
    Se il token non contiene lacune, applica la funzione `normalize_greek` per normalizzare il caso.
        token (str): Il token da processare.
        str: Il token pulito o normalizzato.

    Args:
        token (str): The token to be processed.

    Returns:
        list[str]: Lista di token puliti o normalizzati.
    """
    return clean_lacunae(token).split() if contains_lacunae(token) else token.split()


def clean_text_from_gaps(text: str, case_folding: bool = True) -> str:
    """
    Pulisce il testo dalle lacune, lasciando invariata la punteggiatura.
    Viene usato questo metodo per pulire i testi di addestramento, per poi suddividerlo in frasi.

    Args:
        text (str): testo di addestramento

    Returns:
        clean_text (str): testo pulito dalle lacune
    """
    cleaned_text = clean_text_content(text)
    tokens = clean_tokens(cleaned_text)
    result_text = " ".join(tokens).strip()
    return normalize_greek(result_text) if case_folding else result_text


def process_editorial_marks(text: str) -> str:
    """
    Processa le annotazioni editoriali presenti nel testo.
    """

    def process_integrations(text: str) -> str:
        """
        Rimuove le integrazioni `||` presenti nel testo.

        Args:
        text (str): Il testo da processare.

        Returns:
        str: Il testo processato senza le doppie stanghe `||` e con le parole unite.
        """
        text = text.replace("||", " ") if "||" in text else text
        text = text.replace("‖", " ") if "‖" in text else text
        return text
    
    def process_dactyl_patterns (text : str) -> str:
        """
        Rimuove i pattern che identificano i dattili o sequenze di dattili in versi incompleti presenti nei testi poetici. 

        Args:
        text (str): Il testo da processare.

        Returns:
        str: Il testo senza i modelli di dactilo.
        """
        return COMBINED_DACTYL_PATTERNS.sub("<gap/>", text) if COMBINED_DACTYL_PATTERNS.search(text) else text

    def process_leiden_lb(text: str) -> str:
        """
        Rimuove i *line breaks* di leiden `|` presenti nel testo.

        Args:
        text (str): Il testo da processare.

        Returns:
        str: Il testo senza *line breaks* `|`.
        """
        return text.replace("|", " ") if "|" in text else text
    
    def process_unclear_signs(text: str) -> str:
        """
        Rimuove i segni che rappresentano incertezze ('+' e '*') dal testo.

        Args:
            text (str): Il testo da processare.

        Returns:
            str: Il testo senza i segni '+' e '*'.
        """
        text = text.replace("+", "") if "+" in text else text
        text = text.replace("*", "") if "*" in text else text
        return text
    

    def process_vacat_text(text: str) -> str:
        """
        Rimuove i tag `vac.` e `vacat` dal testo, che rappresentano secondo le convenzioni di Leiden
        del testo mancante non inciso sulla pietra o non presente sul papiro.
        """
        return VACAT_REGEX.sub("", text) if VACAT_REGEX.search(text) else text

    def process_brackets(text: str) -> str:
        """
            Rimuove le parentesi quadre mantenendo il contenuto interno.
            Prima della rimozione viene applicata una sostituzione nel testo per rimuovere i gap di lunghezza sconosciuta che 
            rappresentano frasi.
        """
        return remove_brackets(
            re.sub(
                r"\.\s*<gap/>\s*\.", ".", text
            )  # Rimuove gap di lunghezza sconosciuta che rappresentano frasi
        )

    def process_missing_lines(text: str) -> str:
        """
        Rimuove linee mancanti dal testo.
        """
        return (
            MISSING_LINES_REGEX.sub("", text)
            if MISSING_LINES_REGEX.search(text)
            else text
        )

    def process_parentheses(text: str) -> str:
        """
        Estensione delle abbreviazioni (a(bc)) -> (abc).
        """
        return text.replace("(", "").replace(")", "")

    def process_markers(text: str) -> str:
        """
        Processa aggiunte di testo <abc> -> abc.
        """
        if MARKER_REGEX.search(text):
            text = MARKER_REGEX.sub(r"\1", text)

        if UNKNOWN_LEFT_MARKER_REGEX.search(text):
            text = UNKNOWN_LEFT_MARKER_REGEX.sub(r"<gap/>\1", text)

        if EXTENDED_LINE_RIGHT_MARKER_REGEX.search(text):
            text = EXTENDED_LINE_RIGHT_MARKER_REGEX.sub("", text)

        if "&gt;" in text:
            text = text.replace("&gt;", "")

        if "&lt;" in text:
            text = text.replace("&lt;", "")

        return text

    def process_expunctions(text: str) -> str:
        """
        Rimuove espunzioni {abc} -> "".
        """
        return EXPUNCTION_REGEX.sub("", text) if EXPUNCTION_REGEX.search(text) else text

    def process_parallel_text(text: str) -> str:
        """
        Rimuove testo fornito in parallelo
        """
        text =  OBELISK_REGEX.sub("", text) if OBELISK_REGEX.search(text) else text  # processazione parole morte
        text = text.replace("†", "") if "†" in text else text
        text = text.replace("_", "") if "_" in text else text
        return text

    def process_double_obelisks(text: str) -> str:
        return NOTES_REGEX.sub("", text) if NOTES_REGEX.search(text) else text

    def process_doubts(text: str) -> str:
        """
        Rimuove i punti interrogativi da una stringa di testo.

        Args:
            text (str): La stringa di input da elaborare.

        Returns:
            str: La stringa risultante con i punti interrogativi rimossi.
        """
        return text.replace("?", "")

    def process_dash_if_needed(text: str) -> str:
        return filter_dash(text) if "-" in text else text

    # Applica le varie trasformazioni
    trasformations = [
        process_integrations,
        process_leiden_lb,
        process_unclear_signs,
        process_brackets,
        process_dactyl_patterns,
        process_vacat_text,
        process_double_obelisks,
        process_doubts,
        process_missing_lines,
        process_parentheses,
        process_markers,
        process_expunctions,
        process_parallel_text,
        process_dash_if_needed,
    ]
    for transform in trasformations:
        text = transform(text)

    return text


def clean_text_content(text: str) -> str:
    """
    Applica una serie di pulizie al testo, come la rimozione di parentesi, la gestione dei trattini e la sostituzione di linee mancanti.

    Args:
        text (str): Il testo da pulire.

    Returns:
        str: Il testo pulito.
    """
    return process_editorial_marks(text)


def clean_tokens(text: str) -> list[str]:
    """
    Pulisce e normalizza i token estratti dal testo.

    Args:
        text (str): Il testo da cui estrarre i token, si assume essere "pulito".

    Returns:
        list[str]: Lista di token puliti e normalizzati.
    """

    cleaned_tokens = []
    for token in get_tokens_from_clean_text(text):
        cleaned_tokens.extend(process_token(token))
    return cleaned_tokens
