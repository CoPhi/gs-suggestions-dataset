from backend.core.preprocess import clean_supplements, clean_text_from_gaps
from tqdm import tqdm
from models.ngrams.train import load_abs, split_abs, get_sentences, get_tokens_from_clean_text
from datasets import (
    load_dataset,
    Dataset,
)
from models.bert.finetuning import (
    CHUNK_SIZE,
    BERT_UNK_TOKEN,
    BERT_MAX_SEQ_LENGTH,
    MAX_UNK_TOKEN_TRESHOLD,
    MAX_MASK_TOKEN_TRESHOLD,
    MIN_MASK_TOKEN_TRESHOLD,
    MIN_SENT_TOKEN_TRESHOLD,
    get_model_config,
)

import re

"""UTILS DATASET"""

def get_db():
    return load_dataset("GabrieleGiannessi/maat-corpus")


def generate_mask_tokens(num: int) -> str:
    mask_str = " "
    for _ in range(num):
        mask_str += "[MASK] "
    return mask_str


def load_and_split_sentences(test_size: float = 0.1) -> tuple[list, list]:
    """
    Carica i blocchi anonimi e li divide in due set: uno per l'addestramento e uno per il test.
    I blocchi anonimi vengono caricati dai file JSON presenti nella cartella `data/` e suddivisi in due set: uno per l'addestramento e uno per il test.

        Args:
        test_size (float): La proporzione del set di test rispetto al totale. Default è 0.1 (10%).
    Returns:
        tuple: Due liste di frasi, una per l'addestramento e una per il test.
    """
    abs = load_abs()
    train_abs, test_abs = split_abs(abs, test_size)
    return train_abs, test_abs


def get_sent_from_tokens(tokens: list[str]):
    return " ".join(tokens)


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Divide una lista di frasi in blocchi più piccoli di token con una dimensione specificata.
    Chunking basato su word token, adatto per modelli a n-grammi.

    Argomenti:
        sentences (list[str]): Una lista di frasi da suddividere in blocchi.
        chunk_size (int, opzionale): Il numero di token per blocco. Valore predefinito CHUNK_SIZE.

    Restituisce:
        list[str]: Una lista di blocchi, dove ogni blocco è una stringa contenente `chunk_size` token.
                   Solo i blocchi con una dimensione esatta di `chunk_size` sono inclusi nell'output.
    """
    chunked = []
    for sentence in sentences:
        tokens = sentence.split()
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            if len(chunk) == chunk_size:
                chunked.append(" ".join(chunk))
    return chunked


def chunk_for_bert(sentences: list[str], tokenizer, max_length: int = BERT_MAX_SEQ_LENGTH) -> list[str]:
    """
    Divide una lista di frasi in blocchi rispettando il limite massimo di sub-word token di BERT.
    Le frasi vengono aggregate fino a raggiungere il limite, evitando di spezzare una frase a metà.

    Argomenti:
        sentences (list[str]): Una lista di frasi da suddividere in blocchi.
        tokenizer: Il tokenizer del modello BERT, usato per calcolare la lunghezza in sub-word token.
        max_length (int, opzionale): Il numero massimo di sub-word token per blocco.
                                      Valore predefinito BERT_MAX_SEQ_LENGTH (510 = 512 - [CLS] - [SEP]).

    Restituisce:
        list[str]: Una lista di blocchi, dove ogni blocco rispetta il limite di sub-word token.
                   Le frasi che singolarmente superano il limite vengono troncate.
    """
    chunks = []
    current_chunk_tokens = []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)

        # Se una singola frase supera il limite, la tronchiamo
        if len(sent_tokens) > max_length:
            if current_chunk_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
                current_chunk_tokens = []
                current_length = 0
            chunks.append(tokenizer.convert_tokens_to_string(sent_tokens[:max_length]))
            continue

        # Se aggiungere la frase supera il limite, chiudiamo il chunk corrente
        if current_length + len(sent_tokens) > max_length:
            if current_chunk_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
            current_chunk_tokens = sent_tokens
            current_length = len(sent_tokens)
        else:
            current_chunk_tokens.extend(sent_tokens)
            current_length += len(sent_tokens)

    # Salva l'ultimo chunk
    if current_chunk_tokens:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))

    return chunks


def get_cast_unk_tokens_text(text: str) -> str:
    return text.replace("<UNK>", BERT_UNK_TOKEN)


def get_num_unk_tokens(text: str) -> int:
    c = 0
    for tkn in get_tokens_from_clean_text(text):
        if tkn == BERT_UNK_TOKEN:
            c += 1

    return c


def _filter_sentences(abs: list, remove_punct: bool = True) -> list[str]:
    """
    Logica di filtraggio condivisa per l'estrazione di frasi dai blocchi anonimi.
    Filtra le frasi che non soddisfano i criteri di qualità per il finetuning:
    - Soglia minima di token (MIN_SENT_TOKEN_TRESHOLD)
    - Soglia massima di token sconosciuti (10% della frase)

    La rimozione della punteggiatura è parametrizzata per supportare modelli
    con requisiti diversi:
    - AristoBERTo (GreekBERT): remove_punct=True (tokenizer per greco moderno)
    - GreBerta / Logion: remove_punct=False (tokenizer per greco antico)

    Args:
        abs (list): Lista di blocchi anonimi da cui estrarre le frasi.
        remove_punct (bool): Se True, rimuove la punteggiatura dalle frasi.
                             Default True per retrocompatibilità.

    Returns:
        list[str]: Una lista di frasi filtrate che soddisfano i criteri specificati.
    """
    sentences = []
    for sent_tkns in tqdm(
        get_sentences(abs, case_folding=True, remove_punct=remove_punct),
        desc="Loading set",
        unit="sentence",
        leave=False,
    ):
        processed_text = get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns))
        if (
            get_num_unk_tokens(processed_text) >= (len(sent_tkns) * 0.1)
            or len(sent_tkns) < MIN_SENT_TOKEN_TRESHOLD
        ):
            continue
        sentences.append(processed_text)

    return sentences


def get_processed_sentences(abs: list, remove_punct: bool = True) -> Dataset:
    """
    Estrae e filtra le frasi di addestramento da una lista di blocchi anonimi,
    restituendo un Dataset HuggingFace.

    Args:
        abs (list): Lista di blocchi anonimi da cui estrarre le frasi.
        remove_punct (bool): Se True, rimuove la punteggiatura.

    Returns:
        Dataset: Dataset HuggingFace con colonna 'text'.
    """
    return Dataset.from_dict({"text": _filter_sentences(abs, remove_punct)})


# def get_filtered_processed_sentences(abs: list, remove_punct: bool = True) -> list[str]:
#     """
#     Estrae e filtra le frasi di addestramento da una lista di blocchi anonimi,
#     restituendo una lista di stringhe.

#     Args:
#         abs (list): Lista di blocchi anonimi da cui estrarre le frasi.
#         remove_punct (bool): Se True, rimuove la punteggiatura.

#     Returns:
#         list[str]: Lista di frasi filtrate.
#     """
#     return _filter_sentences(abs, remove_punct)


def get_test_cases_from_abs(abs: list):
    """
    Estrae i test_case dai blocchi anonimi, sostituendo la/e parola/e da predire con [MASK]
    Ci si assicura che le frasi contengono un solo token sconosciuto
    """

    blocks = []

    for ab in tqdm(
        abs,
        desc="Loading test cases",
        unit="test case",
        leave=False,
    ):
        supplements = clean_supplements(ab["training_text"])
        if not supplements:
            continue

        # devo individuare le parole da predire e sostituirle con [MASK]
        for i, obj in enumerate(ab["test_cases"]):

            if i >= len(supplements):
                break

            if (
                len(supplements[i]) == 0
                or (
                    len(supplements[i]) > MAX_MASK_TOKEN_TRESHOLD
                    or len(supplements[i]) < MIN_MASK_TOKEN_TRESHOLD
                )
                or "<UNK>" in supplements[i]
            ):
                continue

            obj["test_case"] = re.sub(
                r"\S*\[(\S*)\]\S*", r"[\1]", obj["test_case"], count=1
            )  # inquadro l'insieme delle parole influenzate dal supplemento
            test_case = get_cast_unk_tokens_text(
                clean_text_from_gaps(obj["test_case"].split("[")[0])
                + generate_mask_tokens(len(supplements[i]))
                + clean_text_from_gaps(obj["test_case"].split("]")[1])
            )

            # Controllo da rifare
            if (
                get_num_unk_tokens(test_case) > MAX_UNK_TOKEN_TRESHOLD
                or len(test_case) > 512
            ):  # prendiamo solo i casi di test con un numero di token sconosciuti inferiori alla soglia per non confondere troppo il modello
                continue

            if (
                len(supplements[i]) > 1
            ):  # se la parola da predire è composta da più di una parola, divido il test_case in più test_case per predire una parola alla volta
                blocks.extend(
                    split_test_case_with_multiple_mask_tokens(
                        obj["test_case"], supplements[i]
                    )
                )
            else:
                blocks.append(
                    {
                        "text": test_case,  # caso di test in cui la/e parola/e da predire sono state sostituite con [MASK]
                        "label": get_sent_from_tokens(
                            supplements[i]
                        ),  # la parola da predire
                    }
                )

    return Dataset.from_dict(
        {
            "text": [test_case["text"] for test_case in blocks],
            "label": [test_case["label"] for test_case in blocks],
        }
    )


def split_test_case_with_multiple_mask_tokens(
    test_case: str, supplements: list[str]
) -> list[dict]:
    """
    Genera una lista di casi di test con un singolo token mascherato e la relativa etichetta,
    sostituendo gli altri token mascherati con le rispettive etichette.

    Args:
        test_case (str): Frase contenente una sequenza mascherata ([MASK]).
        supplements (list[str]): Lista di etichette associate ai token mascherati.

    Returns:
        list[dict]: Lista di dizionari, ciascuno contenente un test_case con un solo token mascherato
        e gli altri token mascherati sostituiti con le rispettive etichette.
    """
    match = re.search(
        r"(\[MASK\](?:\s+\[MASK\])*)", test_case
    )  # Trova la sequenza completa di token [MASK]
    if not match:
        return []  # Nessun token mascherato trovato

    mask_seq = match.group().split()  # Lista dei token [MASK]

    if len(mask_seq) != len(supplements):
        raise ValueError(
            "La lista delle etichette deve avere la stessa lunghezza dei token [MASK]."
        )

    # Genera i test case sostituendo uno per volta ogni [MASK]
    blocks = [
        {
            "test_case": test_case[: match.start()]
            + " ".join(
                supplements[j] if i != j else "[MASK]" for j in range(len(mask_seq))
            )
            + test_case[match.end() :],
            "label": supplements[i],
        }
        for i in range(len(mask_seq))
    ]

    return blocks