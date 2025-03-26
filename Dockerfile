FROM python:3.11.11-slim

WORKDIR /api-models

COPY ./config ./config

COPY ./api ./api

COPY ./train ./train

COPY ./inference ./inference

COPY ./metrics ./metrics

COPY ./utils ./utils

COPY api-requirements.txt api-requirements.txt

RUN pip install -U pip && pip install -r api-requirements.txt

CMD uvicorn api.main:app --reload