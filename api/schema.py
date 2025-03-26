def serial_model(model) -> dict:
    return {
        "ID": str(model["_id"]),
        "LM_SCORE": model["LM_SCORE"],
        "GAMMA": model["GAMMA"],
        "K_PRED": model["K_PRED"],
        "TEST_SIZE": model["TEST_SIZE"],
        "N": model["N"],
        "CORPUS_NAMES": model["CORPUS_NAMES"],
    }

def list_models(models) -> list:
    return [serial_model(model) for model in models]