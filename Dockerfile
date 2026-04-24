FROM python:3.12-slim

WORKDIR /app

# Default dependencies and git (asked by CLTK)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Installing dependencies
RUN pip install uv && \
    uv sync --frozen --no-cache

COPY backend/ ./backend/
COPY models/ ./models/

ENV PATH="/app/.venv/bin:$PATH"

# docker run -v ./data:/app/data gabrielegiannessi/gs-api:latest
VOLUME ["/app/data"]

CMD ["uv", "run", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
