FROM python:3.11.11-slim

WORKDIR /app

# Default dependencies and git (asked by CLTK)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Installing dependencies
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY train/ ./train/
COPY inference/ ./inference/
COPY predictions/ ./predictions/
COPY config/ ./config/
COPY utils/ ./utils/
COPY api/ ./api/
COPY metrics/ ./metrics/

# docker run -v ./data:/app/data gabrielegiannessi/gs-api:latest
VOLUME ["/app/data"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
