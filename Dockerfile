FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Default dependencies and git (asked by CLTK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV UV_HTTP_TIMEOUT=600 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Install only API & inference dependencies (no dev/train groups, CPU-only PyTorch without NVIDIA CUDA libraries)
RUN uv sync --frozen --no-dev --no-cache \
        --no-install-package torch \
        --no-install-package triton \
        --no-install-package cuda-bindings \
        --no-install-package cuda-pathfinder \
        --no-install-package cuda-toolkit \
        --no-install-package nvidia-cublas \
        --no-install-package nvidia-cuda-cupti \
        --no-install-package nvidia-cuda-nvrtc \
        --no-install-package nvidia-cuda-runtime \
        --no-install-package nvidia-cudnn-cu13 \
        --no-install-package nvidia-cufft \
        --no-install-package nvidia-cufile \
        --no-install-package nvidia-curand \
        --no-install-package nvidia-cusolver \
        --no-install-package nvidia-cusparse \
        --no-install-package nvidia-cusparselt-cu13 \
        --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink \
        --no-install-package nvidia-nvshmem-cu13 \
        --no-install-package nvidia-nvtx && \
    uv pip install --no-cache "torch>=2.6.0,<3" --index-url https://download.pytorch.org/whl/cpu

COPY backend/ ./backend/
COPY models/ ./models/

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
