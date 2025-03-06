from typing import Optional
from nltk.lm.api import LanguageModel
from models.training import pipeline_train
from config.settings import LM_TYPES, GAMMAS, K_PREDICTIONS, TEST_SIZES, DIMENSIONS


class ModelService:
    """
    Questa classe gestisce i modelli con cui è possibile creare i suggerimenti da fornire tramite l'API.
    """

    def __init__(self):
        self._model = None  # modello selezionato dal client
        self.models = []  # modelli caricati

    def get_model(self):
        return self._model

    def set_model(self, curr_model):
        self._model = curr_model

    def check_ngrams_model_params(
        self, k_pred: int, lm_type: str, ngrams_order: int, test_size: float, gamma: Optional[float]
    ) -> bool:
        """
        Valida i parametri per il modello n-grams.

        Argomenti:
            k_pred (int): Il numero di previsioni da fare.
            lm_type (str): Il tipo di modello linguistico.
            ngrams_order (int): L'ordine degli n-grams.
            test_size (float): La dimensione del dataset di test.
            gamma (Optional[float]): Il parametro gamma, richiesto se lm_type è "MLE".

        Ritorna:
            bool: True se tutti i parametri sono validi, False altrimenti.
        """
        return (
            lm_type in LM_TYPES
            and k_pred in K_PREDICTIONS
            and ((gamma and lm_type == "MLE") or gamma in GAMMAS)
            and ngrams_order in DIMENSIONS
            and test_size in TEST_SIZES
        )

    def load_ngram_model(
        self,
        k_pred: int, 
        lm_type: str,
        ngrams_order: int,
        test_size: float,
        gamma: Optional[float] = None,
    ):
        """
            Carica un modello di n-grammi con i parametri specificati e lo aggiunge alla lista dei modelli.

            Args:
                k_pred (int): Numero di predizioni da generare.
                lm_type (str): Tipo di modello di linguaggio da utilizzare.
                ngrams_order (int): Ordine degli n-grammi.
                test_size (float): Percentuale del dataset da utilizzare per il test.
                gamma (Optional[float], optional): Parametro opzionale per il modello di linguaggio. Default è None.

            Returns:
                None: Se i parametri del modello di n-grammi non sono validi.
        """ 
        if not self.check_ngrams_model_params(k_pred, lm_type, ngrams_order, test_size, gamma):
            return None

        lm, _ = pipeline_train(
            lm_type=lm_type,
            gamma=gamma,
            n=ngrams_order,
            test_size=test_size,
        )

        model_info = {
            "lm": lm,
            "model": "ngrams",
            "k_pred": k_pred, 
            "lm_type": lm_type,
            "gamma": gamma,
            "ngrams_order": ngrams_order,
            "test_size": test_size,
        }
        self.models.append(model_info)
        self.set_model(model_info)

    def get_ngram_model(self, k_pred: int, lm_type: str, n: int, test_size: float, gamma: Optional[float]):
        """
        Cerchiamo tra i modelli salvati quello con le caratteristiche presenti nei parametri.
        Se non lo troviamo: lo carichiamo e lo settiamo come modello corrente.
        Args:
            k_pred (int): Numero di predizioni da considerare.
            lm_type (str): Tipo di modello linguistico.
            n (int): Ordine degli n-grammi.
            test_size (float): Percentuale del dataset da utilizzare per il test.
            gamma (Optional[float]): Parametro opzionale per la regolarizzazione.
        Returns:
            Il modello n-grammi corrente se trovato, altrimenti None.
        """
        
        if not self.check_ngrams_model_params(k_pred, lm_type, n, test_size, gamma):
            return None
        
        if self._model is not None and self._model["model"] == "ngrams":
            if (
                self._model["lm_type"] == lm_type
                and self._model["k_pred"] == k_pred
                and self._model["ngrams_order"] == n
                and self._model["test_size"] == test_size
                and self._model["gamma"] == gamma
            ):
                return self.get_model()

        for model in [obj for obj in self.models if obj["model"] == "ngrams"]:
            if (
                self._model["k_pred"] == k_pred
                and model["lm_type"] == lm_type
                and model["ngrams_order"] == n
                and model["test_size"] == test_size
                and model["gamma"] == gamma
            ):
                self.set_model(model["lm"])
                return self.get_model()

        self.load_ngram_model(lm_type, n, test_size, gamma)
        return self.get_model()

model_service = ModelService()