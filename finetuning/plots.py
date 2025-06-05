import matplotlib.pyplot as plt
import json
import re


def get_eval_result(model: str) -> dict:
    """
    Carica i risultati di valutazione da un file JSON.

    Args:
        model (str): Il nome del modello per cui caricare i risultati.

    Returns:
        dict: Un dizionario contenente i risultati di valutazione.
    """
    with open(f"finetuning/eval_results_{model}.json", "r") as f:
        result = json.load(f)
        return result


def extract_topK_accuracy(result: dict) -> tuple[list, list]:
    """
    Estrae i valori di top-K accuracy dal dizionario dei risultati.

    Args:
        result (dict): Dizionario con i risultati della valutazione.

    Returns:
        Tuple (list[int], list[float]): Lista di K e accuracies corrispondenti.
    """
    K_list = []
    acc_list = []
    pattern = re.compile(r"eval_top(\d+)_acc")

    for key, value in result.items():
        match = pattern.match(key)
        if match:
            K_list.append(int(match.group(1)))
            acc_list.append(value)

    # Ordina per K crescente
    K_list, acc_list = zip(*sorted(zip(K_list, acc_list)))
    return list(K_list), list(acc_list)


def plot_topK_accuracy(
    title="Top-K Accuracy Curve", save_path="topk_accuracy_plot.png"
):
    aristoberto = get_eval_result("AristoBERTo")
    greBERTa = get_eval_result("greBerta")
    logion = get_eval_result("Logion")
    ngrams = get_eval_result("ngrams")

    plt.figure(figsize=(10, 6))

    for model_name, result, marker in [
        ("gs-AristoBERTo", aristoberto, "o"),
        ("gs-greBERTa", greBERTa, "s"),
        ("gs-Logion", logion, "^"),
        ("Ngrams", ngrams, "D"),
    ]:
        K, acc = extract_topK_accuracy(result)
        plt.plot(K, acc, marker=marker, label=model_name)

    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Top-K Accuracy")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Salva il grafico come immagine
    plt.savefig(save_path)
    print(f"Grafico salvato in: {save_path}")

if __name__ == "__main__":
    plot_topK_accuracy()
