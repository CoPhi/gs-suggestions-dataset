FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Contenimento arene glibc, pool thread CPU/XLA e cache modelli neurale/TTL a runtime
ENV UV_HTTP_TIMEOUT=600 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4" \
    GS_MAX_BERT_MODELS=1 \
    GS_MODEL_IDLE_TTL_SECONDS=600 \
    GS_ITHACA_MICRO_BATCH=4

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
