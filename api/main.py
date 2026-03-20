import tomllib
from pathlib import Path
from fastapi import FastAPI
from api.route import router
from fastapi.middleware.cors import CORSMiddleware


def get_version() -> str:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("project", {}).get("version") or data.get("tool", {}).get(
        "poetry", {}
    ).get("version", "unknown")


app = FastAPI(
    title="gs-api",
    version=get_version(),
    description="""API per GreekSchools. Questa API è progettata per offrire l'accesso a modelli linguistici basati su n-grammi e BERT per la generazione di supplementi testuali. 
    Il suo scopo principale è quello di riempire le lacune presenti nei testi, 
    in particolare nei testi antichi o incompleti, suggerendo parole o frasi che possono essere utilizzate per completare i passaggi mancanti. """,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, include_in_schema=True)
