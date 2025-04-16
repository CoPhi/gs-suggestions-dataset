from utils.preprocess import clean_supplements, clean_text_from_gaps
from tqdm import tqdm
from train import load_abs, split_abs, get_sentences, get_tokens_from_clean_text
from datasets import load_dataset, Dataset
from finetuning import (
    CHUNK_SIZE,
    BERT_UNK_TOKEN,
    MAX_UNK_TOKEN_TRESHOLD,
    MAX_MASK_TOKEN_TRESHOLD,
    MIN_MASK_TOKEN_TRESHOLD,
    LACUNAE_REGEX, 
    BERT_TOKENS_PER_WORD
)
from utils import SUPPLEMENTS_REGEX
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import re

"""UTILS DATASET"""


def get_db():
    return load_dataset("GabrieleGiannessi/maat-corpus")


def generate_mask_tokens(num: int) -> str:
    mask_str = " "
    for _ in range(num):
        mask_str += "[MASK] "
    return mask_str


def load_and_split_sentences(test_size: float = 0.1, dev_size: float = 0.1):
    abs = load_abs()
    temp_abs, test_abs = split_abs(abs, test_size)
    train_abs, dev_abs = split_abs(temp_abs, dev_size)

    return train_abs, dev_abs, test_abs


def get_sent_from_tokens(tokens: list[str]):
    return " ".join(tokens)


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Divide una lista di frasi in blocchi più piccoli di token con una dimensione specificata.

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


def get_cast_unk_tokens_text(text: str) -> str:
    return text.replace("<UNK>", BERT_UNK_TOKEN)


def get_num_unk_tokens(text: str) -> int:
    c = 0
    for tkn in get_tokens_from_clean_text(text):
        if tkn == BERT_UNK_TOKEN:
            c += 1

    return c


def get_processed_sentences(train_abs: list):
    """
    Estrae e filtra le frasi di addestramento da una lista di blocchi anonimi e applica i controlli sui token sconosciuti.
    Args:
        train_abs (list): Lista di abstract da cui estrarre le frasi.
    Returns:
        list: Una lista di frasi filtrate che soddisfano i criteri specificati.
    Note:
        - Le frasi vengono estratte utilizzando la funzione `get_sentences` con la punteggiatura preservata.
        - Le frasi contenenti un numero di token sconosciuti maggiore di `MAX_UNK_TOKEN_TRESHOLD` vengono scartate.
    """
    sentences = []
    for sent_tkns in tqdm(
        get_sentences(train_abs, remove_punct=False),
        desc="Loading train set",
        unit="sentence",
        leave=False,
    ):
        processed_text = get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns))
        if get_num_unk_tokens(processed_text) > MAX_UNK_TOKEN_TRESHOLD:
            continue
        sentences.append(processed_text)

    return Dataset.from_dict(
        {
            "text": sentences,
        }
    )


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
                blocks.extend(split_test_case_with_multiple_mask_tokens(obj["test_case"], supplements[i]))
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


def split_test_case_with_multiple_mask_tokens(test_case: str, supplements: list[str]) -> list[dict]:
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
    match = re.search(r'(\[MASK\](?:\s+\[MASK\])*)', test_case) # Trova la sequenza completa di token [MASK]
    if not match:
        return []  # Nessun token mascherato trovato

    mask_seq = match.group().split()  # Lista dei token [MASK]

    if len(mask_seq) != len(supplements):
        raise ValueError("La lista delle etichette deve avere la stessa lunghezza dei token [MASK].")

    # Genera i test case sostituendo uno per volta ogni [MASK]
    blocks = [
        {
            "test_case": test_case[:match.start()] + " ".join(
                supplements[j] if i != j else "[MASK]" for j in range(len(mask_seq))
            ) + test_case[match.end():],
            "label": supplements[i]
        }
        for i in range(len(mask_seq))
    ]

    return blocks


"""UTILS ACCURACY"""


def get_model(model_checkpoint: str):
    return AutoModelForMaskedLM.from_pretrained(model_checkpoint)


def get_tokenizer(model_checkpoint: str):
    return AutoTokenizer.from_pretrained(model_checkpoint)


def get_masker(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer):
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)


def convert_lacuna_to_masks(text: str, mask_token="[MASK]") -> str:
    """
    Converte le lacune nel testo in token mascherati.

    Args:
        text (str): Il testo contenente lacune da convertire.
        tokenizer (AutoTokenizer): Il tokenizer utilizzato per determinare i token.
        mask_token (str, opzionale): Il token mascherato da utilizzare. Default è "[MASK]".

    Returns:
        str: Il testo con le lacune sostituite dai token mascherati.
    """
    def get_mask_sequence(match)-> str:
         dots = match.group(1)
         return "".join([mask_token] * int(len(dots) / BERT_TOKENS_PER_WORD))
    
    gap_matches = list(SUPPLEMENTS_REGEX.finditer(text)) 
    if len(gap_matches) != 1:
        return 
    
    seq = gap_matches[0]
    start, end = seq.start(), seq.end()
    
    attached_left = start > 0 and text[start - 1].isalpha()
    attached_right = end < len(text) and text[end + 1].isalpha()
    
    converted_text = re.sub(
        LACUNAE_REGEX,
        lambda match: get_mask_sequence(match),
        text,
    )
    
    return (converted_text, attached_left, attached_right)
    
    