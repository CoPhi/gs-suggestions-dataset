FROM python:3.11.11-slim

WORKDIR /app

# Installa git e dipendenze di base (richiesto da CLTK)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY api-requirements.txt api-requirements.txt
RUN pip install -U pip && pip install -r api-requirements.txt

COPY ./config ./config
COPY ./api ./api
COPY ./train ./train
COPY ./inference ./inference
COPY ./metrics ./metrics
COPY ./utils ./utils
COPY ./data ./data

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]