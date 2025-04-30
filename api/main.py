from fastapi import FastAPI
from api.route import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="gs-api",
    version="0.1.0",
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
