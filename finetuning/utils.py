from utils.preprocess import clean_supplements, clean_text_from_gaps
from tqdm import tqdm
from datasets import load_dataset
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


def get_test_cases_from_abs(abs: list):
    """
    Estrae i test_case dai blocchi anonimi, sostituendo la/e parola/e da predire con [MASK]
    Ci si assicura che le frasi contengono un solo token sconosciuto
    """

    test_set = []

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
            
            if len(supplements[i]) == 0: 
                continue
            
            obj["test_case"] = re.sub(
                r"\S*\[(\S*)\]\S*", r"[\1]", obj["test_case"], count=1
            )  # inquadro l'insieme delle parole influenzate dal supplemento
            test_case = (
                clean_text_from_gaps(obj["test_case"].split("[")[0])
                + generate_mask_tokens(len(supplements[i]))
                + clean_text_from_gaps(obj["test_case"].split("]")[1])
            )
            
            if (
                "<UNK>" in test_case
            ):  # prendiamo solo i casi di test senza token sconosciuti per non confondere il modello
                continue
            
            if len(supplements[i]) > 1: # se la parola da predire è composta da più di una parola, divido il test_case in più test_case per predire una parola alla volta
                suppl_str = " "
                for supplement in supplements[i]:
                    single_mask_test_case = (
                        clean_text_from_gaps(obj["test_case"].split("[")[0])
                        + suppl_str + "[MASK] "
                        + clean_text_from_gaps(obj["test_case"].split("]")[1])
                    )
                    suppl_str += supplement + " "
                    test_set.append(
                        {
                            "text": single_mask_test_case,  # caso di test in cui la/e parola/e da predire sono state sostituite con [MASK]
                            "gold_label": [supplement],  # la/e parola/e da predire
                        }
                    )
            else: 
                test_set.append(
                    {
                        "text": test_case,  # caso di test in cui la/e parola/e da predire sono state sostituite con [MASK]
                        "gold_label": supplements[i],  # la parola da predire
                    }
                )

    return test_set


"""UTILS ACCURACY"""


def get_model(checkpoint: str):
    return AutoModelForMaskedLM.from_pretrained(checkpoint)


def get_tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint)


def get_masker(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer):
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)
