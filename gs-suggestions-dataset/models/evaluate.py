import re
from tqdm import tqdm

from cltk.core.data_types import Doc

from nltk import ngrams
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import (
    pad_both_ends,
    flatten,
)

from utils.preprocess import clean_text
from models.training import get_sentences, load_lm
from config.settings import tokenizer, sentence_tokenizer, K_PRED, BATCH_SIZE, N

lm, test_abs = load_lm()


def evaluate(lm: LanguageModel, test_abs: list) -> float:
    """
    Funzione di valutazione del modello.
    Calcola la perplessità su dei dati di valutazione o sul test set.

    Returns:
        float: Perplessità.
    """

    test_ngrams = []
    for sentence in get_sentences(test_abs):
        test_ngrams.extend(
            list(
                ngrams(
                    sentence,
                    N,
                    pad_left=True,
                    pad_right=True,
                    left_pad_symbol="<s>",
                    right_pad_symbol="</s>",
                )
            )
        )

    if not lm.vocab:
        raise ValueError("Il modello non è stato addestrato correttamente.")
    try:
        return lm.perplexity(test_ngrams)
    except ZeroDivisionError:
        raise RuntimeError(
            "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
        )


def accuracy(lm: LanguageModel, test_abs: list) -> float:
    """
    Calcola l'accuratezza del modello sui dati forniti (abs) in batch.

    Args:
        abs (list): Lista di anonymous block per il calcolo dell'accuratezza.

    Returns:
        float: Accuratezza del modello.
    """
    correct_predictions = 0
    total_predictions = 0

    for start in tqdm(
        range(0, len(test_abs), BATCH_SIZE),
        desc="Calcolo accuracy",
        unit="ab",
        leave=False,
    ):
        batch = test_abs[start : start + BATCH_SIZE]  # batch di blocchi anonimi
        for ab in batch:
            if ab["language"] == "grc":
                supplements = list(
                    map(
                        str.strip,
                        [
                            re.sub(r"[\[\]]", "", suppl)
                            for suppl in re.findall(
                                r"\w*\[[^\]]+\]\w*",
                                re.sub(r"<gap/>", " ", ab["training_text"]),
                            )
                        ],
                    )
                )
                if not supplements:
                    continue  # non ci sono blocchi da predire

                for i, obj in enumerate(ab["test_cases"]):
                    if i >= len(supplements):
                        break

                    test_case = obj["test_case"]

                    supplement_words = tokenizer.run(
                        input_doc=Doc(raw=clean_text(supplements[i]))
                    ).tokens

                    context = list(
                        flatten(
                            [
                                list(
                                    pad_both_ends(
                                        tokenizer.run(input_doc=Doc(raw=sent)).tokens,
                                        n=N,
                                    )
                                )
                                for sent in sentence_tokenizer.tokenize(
                                    clean_text(
                                        re.sub(r"[^\s]+\[", "[", test_case).split("[")[
                                            0
                                        ]
                                    )
                                )
                            ]
                        )
                    )[:-(N-1)]

                    predictions = []  # qui salvo le k predizioni del modello

                    dist_words_context = sorted(
                        lm.context_counts(lm.vocab.lookup(context[-1:])).items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )  # distribuzione di frequenze per le parole, dato il contesto

                    for word, _ in dist_words_context[:K_PRED]:
                        temp_context = context + [word]  # contesto adattato
                        current_prediction = [word]
                        while len(current_prediction) < len(supplement_words):
                            token = lm.generate(
                                text_seed=temp_context[-(N-1):], num_words=1
                            )
                            current_prediction.append(token)
                            context.append(token)

                        predictions.append(current_prediction)

                    if supplement_words in predictions:
                        correct_predictions += 1

                    total_predictions += 1

    return round((correct_predictions / total_predictions) * 100, 2)


if __name__ == "__main__":
    lm, test_abs = load_lm()  # carico il modello
    print(accuracy(lm, test_abs))
