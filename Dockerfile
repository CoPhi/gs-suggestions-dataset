FROM python:3.11.11-slim

WORKDIR /app

# Installa git e dipendenze di base (richiesto da CLTK)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . . 

#Si installano le dipendenze
RUN pip install -r requirements.txt

#Si espone la porta 
EXPOSE 8000 

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]