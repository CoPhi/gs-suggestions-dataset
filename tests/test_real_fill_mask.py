import torch
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models.bert.inference.predict import fill_mask
from models.bert.evaluation.metrics import evaluate_topK_text
from models.bert.dataset.dev_set import DevCase


def test_simulation():
    checkpoint = "CNR-ILC/gs-GreBerta"
    print(f"--- Inizio Simulazione Refactored [{checkpoint}] ---")

    try:
        # 1. Caricamento Tokenizer reale
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        model = AutoModelForMaskedLM.from_pretrained(checkpoint)

        # 3. Caso di test (Gold con sigma finale 'ς')
        case = DevCase(
            x="Οὐ μὴν ἀπο βιαστέον γε τοῦτ' ἐστιν διὰ τῶν κατὰ τὰς ἑρμηνείας συν ηθειῶν καὶ ταῦτα μηθὲν ἐν δεικνυμένους περὶ τῆς τοῦ σοφοῦ κτήσεώς τε καὶ χρή σεως, ὥσπερ οἱ σοφισταὶ ποι οῦσιν, ἀλλ' ἀνάγοντας ἐπὶ τὴν ὑπάρχουσαν ἡμῖν πρόληψιν περὶ ἀγαθοῦ χρηματιστοῦ, σκεπτέον τε ἐν τίνι τὸ προ ειλημμένον καὶ τῶι πῶς χρη ματιζομένωι, καὶ ὧι ἂν ἐκεῖ ν' ἐπιμαρτυρῆται, κατηγορη τέον τούτου χρηματιστὴν ἀγαθόν· διόπερ εἰ μὲν βου λόμεθα λέγειν ἐν προλήψει τοῦτον ἀγαθὸν χρηματιστὴν τὸν κατὰ τὸ συμφέρον κτώ μενον καὶ ἐπιμελόμενον πλούτου, τὸν σοφὸν μάλιστα τοιοῦτον εἶναι ῥητέον· εἰ δὲ μᾶλλον ἐπὶ τὸν δυνατῶς καὶ ἐντέχνως πολλὰ πορι ζόμενον καὶ μήτε αἰσχρῶς ἐννόμως τε φέρομεν ἐν προλήψει τὸν ἀγαθὸν χρη ματιστήν, κἂν ὅτι μάλιστα πλεῖον κακοπαθῆ κτώμε νος [....]ς ἤπερ ἥδηται, μᾶλ λον ἄλλους τῶν σοφῶν φατέ ον. Οὐθὲν γὰρ ἀφαιρουμένη τοῦ σοφοῦ ἡ τοιαύτη κατη <UNK> μόνον μη <UNK> ἡ γιγνομέ νη κατὰ τὸ συμφέρον κτή σει τε καὶ οἰκονομίαι πλού του. Τῶι γὰρ μὴ ὁρᾶν περὶ τοῦ ὅπως προεστῶτας χρη μάτων ἀκολουθεῖ τὸ συμ φέρον ζηλοῦμεν τοὺς πολ λὰ καὶ ταχέως κτωμένους ἡγούμενοι τούτοις ὑπάρ χειν τὸν λυσιτελῆ τῶι βίωι χρηματισμόν. Οἱ δὲ φιλοσο φεῖν φάσκοντες, ἐξὸν λέγειν ἡμῖν παρ' ἃς αἰτίας ὁ σοφὸς ἐ π' ὠφελίαι μάλιστα καὶ κτή",
            y=["οὕτως"],
            gap_length=4,
            corpus_id="DCLP",
            file_id="62471",
        )

        # 4. Generazione suggerimenti tramite fill_mask
        # La funzione fill_mask userà internamente normalize_greek con case_folding="fold"
        suggestions = fill_mask(
            text=case.x,
            checkpoint=checkpoint,
            model=model,
            tokenizer=tokenizer,
            n_chars=case.gap_length,
            K=20,
        )

        print("\nSuggerimenti generati (foldati):")
        for i, (s, p) in enumerate(suggestions):
            print(f"{i+1}. {s} (score: {p:.4f})")
            # Verifica codici esadecimali per il sigma
            if "σ" in s:
                print(f"   [INFO] Trovato sigma mediano (\\u03c3) nella predizione")

        # 5. Valutazione con metrics.py
        # Anche evaluate_topK_text usa internamente fold
        predictions_batch = [suggestions]
        gold_labels_batch = [case.y]

        metrics = evaluate_topK_text(predictions_batch, gold_labels_batch)

        print(f"\nMetriche: {metrics}")

        if metrics["top1"] == 100.0:
            print(
                "\n✅ VERIFICA RIUSCITA: Il sistema riconosce 'αγαθοσ' come match per 'αγαθος'."
            )
        else:
            print("\n❌ VERIFICA FALLITA: Mismatch tra predizione e gold.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Errore durante la simulazione: {e}")


if __name__ == "__main__":
    test_simulation()
