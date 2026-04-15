import pytest
import torch
import asyncio
from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.bert.inference.predict import fill_mask
from backend.core.preprocess import normalize_greek
from backend.api.services.suggestions_service import SuggestionsService
from enum import Enum

# Fixtures for loading model and tokenizer once
@pytest.fixture(scope="module")
def bert_resources():
    model_name = "bowphs/GreBerta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer

def test_normalize_greek_logic():
    """Verifica che normalize_greek gestisca correttamente i diacritici e il casing."""
    text = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    normalized = normalize_greek(text, case_folding=True)
    # Dovrebbe essere tutto maiuscolo e senza puntini sottoscritti
    assert normalized == "ΑΛΛΑ ΜΗΝ ΕΝ ΤΩΙ ΚΑΤ[.]ΣΚΕΥΑΖΕΙΝ"

def test_fill_mask_auto_detection(bert_resources):
    """Verifica che fill_mask rilevi automaticamente n_chars dalla lacuna [.....]."""
    model, tokenizer = bert_resources
    # Test con lacuna di 5 caratteri
    test_text = "Ἀριστοτέλης ἐστὶν ὁ [.....]ξαν ἀληθῆ φαίνε"
    
    # Chiamata senza n_chars esplicito
    suggestions = fill_mask(
        text=test_text,
        model=model,
        tokenizer=tokenizer,
        K=1,
        normalize_probs=True
    )
    
    assert len(suggestions) > 0
    assert isinstance(suggestions[0][0], str)
    assert 0 <= suggestions[0][1] <= 1.0

def test_fill_mask_normalization_impact(bert_resources):
    """Verifica che l'input normalizzato produca risultati coerenti (GreBerta fix)."""
    model, tokenizer = bert_resources
    text = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    
    # Risultati con testo raw (con diacritici che GreBerta non capisce bene)
    res_raw = fill_mask(text=text, model=model, tokenizer=tokenizer, K=5, normalize_probs=True)
    
    # Risultati con testo normalizzato
    norm_text = normalize_greek(text, case_folding=True)
    res_norm = fill_mask(text=norm_text, model=model, tokenizer=tokenizer, K=5, normalize_probs=True)
    
    # Il miglior suggerimento normalizzato dovrebbe essere 'Α' (per κατασκευάζειν)
    # Mentre quello raw storicamente falliva o dava punteggi bassi/erratici
    top_norm_token = res_norm[0][0]
    assert "Α" in top_norm_token or "Ε" in top_norm_token # "Α" è il target filosofico, "Ε" è grammaticalmente plausibile

@pytest.mark.asyncio
async def test_suggestions_service_bert_flow():
    """Verifica il flusso completo del SuggestionsService con mock per i modelli BERT."""
    # Mock per PredictionCount enum
    class MockCount(Enum):
        FIVE = 5
        @property
        def value(self):
            return 5

    service = SuggestionsService(db_collection=None, gridfs=None)
    
    # Mock dei metodi interni che richiedono DB o HF Hub
    service._validate_hf_checkpoint = lambda x: asyncio.sleep(0)
    service._fetch_model = lambda x: asyncio.sleep(0) or {
        "TYPE": "BERT",
        "CHECKPOINT": "CNR-ILC/gs-GreBerta"
    }
    
    context = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    
    # Eseguiamo la predizione
    # Nota: _predict_bert scarica il modello se non è in cache, pytest-asyncio lo gestisce
    predictions = await service._predict_bert(
        model={"CHECKPOINT": "CNR-ILC/gs-GreBerta"}, 
        context=context, 
        num_predictions=MockCount.FIVE
    )
    
    assert len(predictions) == 5
    for p in predictions:
        assert "sentence" in p
        assert "token_str" in p
        assert "score" in p
        # La lacuna deve essere stata riempita mantenendo il prefisso κατ̣ e il suffisso ς̣κευαζειν
        assert "κατ̣" in p["sentence"]
        assert "ς̣κευαζειν" in p["sentence"]

if __name__ == "__main__":
    # Permette di eseguire il file anche come script
    pytest.main([__file__])
