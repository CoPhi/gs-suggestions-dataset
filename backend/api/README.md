# GreekSchools API (`backend/api`)

Modulo API FastAPI per il servizio di generazione di supplementi testuali e restauro di testi per il progetto GreekSchools.

L'API consente di registrare modelli linguistici nel database MongoDB e di interrogarli tramite endpoint di predizione.

---

## Modelli Supportati

1. **N-grammi**: Modelli n-gram con smoothing MLE o Lidstone (`TYPE`: `"Ngrams"`).
2. **BERT**: Modelli Masked Language Modeling basati su Transformer (es. `CNR-ILC/gs-aristoBERTo`, `CNR-ILC/gs-GreBerta`, `CNR-ILC/gs-Logion`) (`TYPE`: `"BERT"`).
3. **Ithaca**: Modello BigBird character-level fine-tunato sul corpus TLG (es. `CNR-ILC/gs-ithaca-tlg` o checkpoint locale `checkpoints/ithaca/checkpoint_tlg.pkl`) con decodifica Hammersley-Clifford-Besag (HCB) beam search (`TYPE`: `"Ithaca"`).

---

## Guida: Creazione e Utilizzo di Ithaca per l'Inferenza

### 1. Creazione e Memorizzazione del Modello (`POST /models`)

Inviando una richiesta `POST /models` con il payload appropriato, il modello viene registrato nel database MongoDB (`models_collection`). Se non già presente in locale o in cache, il checkpoint Hugging Face Hub viene risolto e reso disponibile per tutte le successive chiamate di inferenza.

#### Payload JSON:
```json
{
  "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
  "TYPE": "Ithaca"
}
```

#### Esempio con cURL:
```bash
curl -X POST "http://localhost:8000/models" \
     -H "Content-Type: application/json" \
     -d '{
       "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
       "TYPE": "Ithaca"
     }'
```

#### Risposta (`201 Created`):
```json
{
  "ID": "67066928e4414eb1d8109bf1"
}
```
> Il valore `ID` restituito è l'identificativo univoco MongoDB (`_id`) da utilizzare nelle chiamate di inferenza.

---

### 2. Verifica dei Modelli Memorizzati (`GET /models` o `GET /models/{id}`)

#### Esempio con cURL:
```bash
curl -X GET "http://localhost:8000/models/67066928e4414eb1d8109bf1"
```

#### Risposta (`200 OK`):
```json
{
  "model": {
    "_id": "67066928e4414eb1d8109bf1",
    "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
    "TYPE": "Ithaca"
  }
}
```

---

### 3. Generazione di Predizioni / Inferenza (`GET /predictions`)

Una volta creato, il modello può essere interrogato passando il `model_id` e il contesto contenente una lacuna indicata tra parentesi quadre nel formato Leiden `[...]` (o trattini `[---]`):

#### Esempio con cURL:
```bash
curl -G "http://localhost:8000/predictions" \
     --data-urlencode "model_id=67066928e4414eb1d8109bf1" \
     --data-urlencode "context=ΚΑΙ ΤΩΝ ΦΙΛΟΣΟΦ[...] ΕΝ ΤΗΙ ΠΟΛΕΙ" \
     --data-urlencode "num_predictions=5"
```

#### Risposta (`200 OK`):
```json
{
  "predictions": [
    {
      "sentence": "ΚΑΙ ΤΩΝ ΦΙΛΟΣΟΦΩΝ ΕΝ ΤΗΙ ΠΟΛΕΙ",
      "token_str": "ΩΝ",
      "score": 0.8421
    },
    {
      "sentence": "ΚΑΙ ΤΩΝ ΦΙΛΟΣΟΦΙΑ ΕΝ ΤΗΙ ΠΟΛΕΙ",
      "token_str": "ΙΑ",
      "score": 0.0814
    }
  ]
}
```

---

### Esempio Completo in Python

```python
import httpx

API_URL = "http://localhost:8000"

with httpx.Client(base_url=API_URL, timeout=60.0) as client:
    # 1. Registra / Crea il modello Ithaca
    create_resp = client.post("/models", json={
        "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
        "TYPE": "Ithaca"
    })
    
    if create_resp.status_code == 201:
        model_id = create_resp.json()["ID"]
        print(f"Modello creato con ID: {model_id}")
    elif create_resp.status_code == 409:
        # Modello già esistente nel db: recuperiamo l'ID
        models = client.get("/models").json()["models"]
        model_id = next(
            m["_id"] for m in models 
            if m.get("TYPE") == "Ithaca" and m.get("CHECKPOINT") == "CNR-ILC/gs-ithaca-tlg"
        )
        print(f"Modello già presente con ID: {model_id}")
    else:
        create_resp.raise_for_status()

    # 2. Esegui l'inferenza
    context = "ΚΑΙ ΤΩΝ ΦΙΛΟΣΟΦ[...] ΕΝ ΤΗΙ ΠΟΛΕΙ"
    pred_resp = client.get("/predictions", params={
        "model_id": model_id,
        "context": context,
        "num_predictions": 5
    })
    pred_resp.raise_for_status()

    predictions = pred_resp.json()["predictions"]
    for idx, pred in enumerate(predictions, 1):
        print(f"{idx}. {pred['token_str']} (confidenza: {pred['score']:.4f}) -> {pred['sentence']}")
```
