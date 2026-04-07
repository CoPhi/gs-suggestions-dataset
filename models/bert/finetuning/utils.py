"""
Utility per il finetuning di modelli BERT su testi in greco antico.

Il modulo espone due livelli di pipeline:
  1. build_raw_dataset   – produce un Dataset grezzo (markup editoriale rimosso,
                           diacritici e punteggiatura preservati) uguale per tutti
                           i modelli.
  2. prepare_dataset_for_model – normalizza e tokenizza il dataset grezzo secondo
                                 la configurazione specifica del checkpoint BERT.

Flusso consigliato
------------------
  raw = build_raw_dataset(abs)
  tokenized_train = prepare_dataset_for_model(raw["train"], checkpoint)
  tokenized_dev   = prepare_dataset_for_model(raw["dev"],   checkpoint)
"""

from functools import partial

from tqdm import tqdm

from backend.core.cleaner import (
    load_abs,
    split_abs_herc_dev,
    get_sentences,
    get_tokens_from_clean_text,
)
from backend.core.preprocess import (
    process_editorial_marks,
    strip_diacritics,
    normalize_greek,
    remove_punctuation,
    clean_tokens,
)
from cltk.alphabet.grc.grc import normalize_grc
from datasets import Dataset, DatasetDict, load_dataset
from models.bert.finetuning import (
    CHUNK_SIZE,
    BERT_UNK_TOKEN,
    BERT_MAX_SEQ_LENGTH,
    MIN_SENT_TOKEN_TRESHOLD,
    get_model_config,
)
from transformers import AutoTokenizer

# Helpers di base


def get_db() -> DatasetDict:
    """Carica il dataset MAAT dal HuggingFace Hub."""
    return load_dataset("GabrieleGiannessi/maat-corpus")


def generate_mask_tokens(num: int) -> str:
    """Restituisce una stringa con *num* token [MASK] separati da spazio."""
    return (" [MASK]" * num).lstrip()


def get_sent_from_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def get_cast_unk_tokens_text(text: str) -> str:
    """Sostituisce il tag <UNK> interno con il token speciale BERT [UNK]."""
    return text.replace("<UNK>", BERT_UNK_TOKEN)


def get_num_unk_tokens(text: str) -> int:
    """Conta i token [UNK] presenti in *text*."""
    return sum(1 for tkn in get_tokens_from_clean_text(text) if tkn == BERT_UNK_TOKEN)


# Caricamento e split del corpus


def load_train_and_dev_set(test_size: float = 0.1) -> tuple[list, list]:
    """
    Carica i blocchi anonimi e li suddivide in train e dev set.

    Il dev set contiene esclusivamente blocchi dei Papiri di Ercolano
    (P.Herc.), garantendo la stratificazione per dominio.

    Args:
        test_size: Proporzione del dev set rispetto ai soli blocchi P.Herc.

    Returns:
        (train_abs, dev_abs)
    """
    abs_ = load_abs()
    return split_abs_herc_dev(abs_, test_size)


# Chunking


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Divide le frasi in blocchi di *chunk_size* word token (per modelli n-grammi).

    I blocchi che non raggiungono la dimensione esatta vengono scartati.

    Args:
        sentences:  Lista di frasi già normalizzate.
        chunk_size: Numero di word token per blocco.

    Returns:
        Lista di stringhe, ciascuna contenente esattamente *chunk_size* token.
    """
    chunked = []
    for sentence in sentences:
        tokens = sentence.split()
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            if len(chunk) == chunk_size:
                chunked.append(" ".join(chunk))
    return chunked


def chunk_for_bert(
    sentences: list[str],
    tokenizer,
    max_length: int = BERT_MAX_SEQ_LENGTH,
) -> list[str]:
    """
    Aggrega frasi in blocchi che rispettano il limite di sub-word token di BERT.

    Le frasi che singolarmente superano *max_length* vengono troncate.

    Args:
        sentences:  Lista di frasi normalizzate.
        tokenizer:  Tokenizer HuggingFace del modello target.
        max_length: Limite di sub-word token (default 510 = 512 − [CLS] − [SEP]).

    Returns:
        Lista di stringhe pronte per la tokenizzazione finale.
    """
    chunks: list[str] = []
    current_tokens: list[str] = []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)

        if len(sent_tokens) > max_length:
            if current_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
                current_tokens = []
                current_length = 0
            chunks.append(tokenizer.convert_tokens_to_string(sent_tokens[:max_length]))
            continue

        if current_length + len(sent_tokens) > max_length:
            if current_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
            current_tokens = sent_tokens
            current_length = len(sent_tokens)
        else:
            current_tokens.extend(sent_tokens)
            current_length += len(sent_tokens)

    if current_tokens:
        chunks.append(tokenizer.convert_tokens_to_string(current_tokens))

    return chunks


# Dataset grezzo (model-agnostic)


def _is_quality_sentence_word_level(
    tokens: list[str],
    unk_ratio_threshold: float = 0.1,
) -> bool:
    """
    Verifica i criteri di qualità su word token (usato per il dataset grezzo).

    Args:
        tokens:               Lista di word token della frase.
        unk_ratio_threshold:  Frazione massima di token [UNK] consentita.

    Returns:
        True se la frase supera entrambe le soglie.
    """
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False
    text = get_sent_from_tokens(tokens)
    unk_count = get_num_unk_tokens(get_cast_unk_tokens_text(text))
    return unk_count < len(tokens) * unk_ratio_threshold


def build_raw_dataset(abs_: list) -> Dataset:
    """
    Produce il dataset grezzo: markup editoriale rimosso, lacune → [UNK],
    diacritici e punteggiatura **preservati**.

    Questa rappresentazione è model-agnostic: può essere caricata una sola
    volta dal Hub e poi normalizzata on-the-fly per ogni modello BERT.

    Args:
        abs_: Lista di blocchi anonimi (già filtrati per language == "grc").

    Returns:
        Dataset HuggingFace con colonna 'text'.
    """
    sentences: list[str] = []
    for sent_tkns in tqdm(
        # case_folding=False e remove_punct=False → testo grezzo
        get_sentences(abs_, case_folding=False, remove_punct=False),
        desc="Building raw dataset",
        unit="sentence",
        leave=False,
    ):
        if not _is_quality_sentence_word_level(sent_tkns):
            continue
        text = get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns))
        sentences.append(text)

    return Dataset.from_dict({"text": sentences})


# Normalizzazione e tokenizzazione model-specific


def _normalize_example(example: dict, config: dict) -> dict:
    """
    Applica la normalizzazione model-specific a un singolo esempio del Dataset.

    Chiamata tramite Dataset.map(); non deve essere invocata direttamente.

    Args:
        example: Dizionario con chiave 'text'.
        config:  Configurazione del modello (da get_model_config).

    Returns:
        Dizionario con chiave 'text' normalizzata.
    """
    text = example["text"]

    if config["strip_diacritics"]:
        text = strip_diacritics(text)

    text = normalize_grc(text)
    text = text.upper() if config["strip_diacritics"] else text.lower()

    if config["remove_punct"]:
        text = remove_punctuation(text)

    text = text.replace("<UNK>", BERT_UNK_TOKEN)
    return {"text": text}


def _tokenize_example(example: dict, tokenizer) -> dict:
    """
    Tokenizza un esempio con il tokenizer del modello specifico.

    Chiamata tramite Dataset.map() dopo _normalize_example; il padding è
    delegato al DataCollatorForLanguageModeling durante il training.

    Args:
        example:   Dizionario con chiave 'text' già normalizzata.
        tokenizer: Tokenizer HuggingFace del modello target.

    Returns:
        Dizionario con 'input_ids' e 'attention_mask'.
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )


def _quality_filter_subword(example: dict, tokenizer) -> bool:
    """
    Verifica i criteri di qualità su sub-word token reali del modello.

    A differenza di _is_quality_sentence_word_level, opera sui sub-word token
    prodotti dal tokenizer specifico, garantendo coerenza con il vocabolario
    del modello.

    Args:
        example:   Dizionario con chiave 'text' già normalizzata.
        tokenizer: Tokenizer HuggingFace del modello target.

    Returns:
        True se la frase supera entrambe le soglie.
    """
    tokens = tokenizer.tokenize(example["text"])
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False
    unk_count = tokens.count(tokenizer.unk_token)
    return unk_count / max(len(tokens), 1) < 0.1


def prepare_dataset_for_model(
    raw_dataset: Dataset,
    checkpoint: str,
    num_proc: int = 4,
) -> Dataset:
    """
    Pipeline completa per un modello BERT specifico:
      1. Normalizzazione (case folding, diacritici, punteggiatura)
      2. Filtraggio qualitativo su sub-word token reali
      3. Tokenizzazione con il tokenizer del checkpoint

    Args:
        raw_dataset: Dataset grezzo prodotto da build_raw_dataset().
        checkpoint:  Es. "CNR-ILC/gs-aristoBERTo".
        num_proc:    Numero di processi paralleli per Dataset.map().

    Returns:
        Dataset con colonne 'input_ids' e 'attention_mask'.
    """
    config = get_model_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    normalized = raw_dataset.map(
        partial(_normalize_example, config=config),
        desc=f"Normalizing [{checkpoint}]",
        num_proc=num_proc,
    )

    filtered = normalized.filter(
        partial(_quality_filter_subword, tokenizer=tokenizer),
        desc=f"Filtering [{checkpoint}]",
        num_proc=num_proc,
    )

    tokenized = filtered.map(
        partial(_tokenize_example, tokenizer=tokenizer),
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing [{checkpoint}]",
        num_proc=num_proc,
    )

    return tokenized


# ---------------------------------------------------------------------------
# Retrocompatibilità
# ---------------------------------------------------------------------------


def get_processed_sentences(
    abs_: list,
    remove_punct: bool = True,
    case_folding: bool = True,
) -> Dataset:
    """
    .. deprecated::
        Usare build_raw_dataset() + prepare_dataset_for_model() per separare
        la costruzione del dataset dalla normalizzazione model-specific.

    Estrae e filtra le frasi da *abs_* restituendo un Dataset HuggingFace.
    """
    sentences: list[str] = []
    for sent_tkns in tqdm(
        get_sentences(abs_, case_folding=case_folding, remove_punct=remove_punct),
        desc="Loading set",
        unit="sentence",
        leave=False,
    ):
        processed_text = get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns))
        if (
            get_num_unk_tokens(processed_text) >= len(sent_tkns) * 0.1
            or len(sent_tkns) < MIN_SENT_TOKEN_TRESHOLD
        ):
            continue
        sentences.append(processed_text)

    return Dataset.from_dict({"text": sentences})
