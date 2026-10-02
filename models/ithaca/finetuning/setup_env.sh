#!/usr/bin/env bash
# ==============================================================================
# Setup automatico dell'ambiente e delle dipendenze per Ithaca su macchina remota
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "CONFIGURAZIONE AMBIENTE ITHACA PER GREEKSCHOOLS"

# 1. Configurazione Variabili d'Ambiente JAX per prevenire conflitti VRAM con PyTorch
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "[1/5] Variabili d'ambiente XLA configurate (no pre-allocation, max 40% VRAM)."

# 2. Creazione cartelle necessarie
mkdir -p packages checkpoints/ithaca data/ithaca eval/results

# 3. Download repository ufficiale Ithaca se non presente
ITHACA_ENGINE_DIR="packages/ithaca_engine"
if [ ! -d "$ITHACA_ENGINE_DIR" ]; then
    echo "[2/5] Clonazione repository google-deepmind/ithaca in $ITHACA_ENGINE_DIR..."
    git clone https://github.com/google-deepmind/ithaca.git "$ITHACA_ENGINE_DIR"
else
    echo "[2/5] Repository Ithaca già presente in $ITHACA_ENGINE_DIR."
fi

# 4. Installazione dipendenze JAX, Flax, Optax
echo "[3/5] Installazione e sincronizzazione dipendenze JAX/Flax/Optax via uv..."
JAX_PKG="jax"
if command -v nvidia-smi &> /dev/null; then
    echo "Rilevata GPU NVIDIA! Configurazione con supporto CUDA 12..."
    JAX_PKG="jax[cuda12]"
fi

if command -v uv &> /dev/null; then
    uv add "$JAX_PKG" "flax" "optax" "chex"
    uv pip install -e "$ITHACA_ENGINE_DIR" --no-deps || true
else
    pip install "$JAX_PKG" "flax" "optax" "chex"
    pip install -e "$ITHACA_ENGINE_DIR" --no-deps || true
fi

# 5. Download del Checkpoint Pre-Addestrato Ufficiale (checkpoint_v1.pkl)
CHECKPOINT_FILE="checkpoints/ithaca/checkpoint_v1.pkl"
CHECKPOINT_URL="https://storage.googleapis.com/ithaca-resources/models/checkpoint_v1.pkl"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "[4/5] Download del checkpoint ufficiale Ithaca da Google Cloud Storage..."
    curl -L --progress-bar -o "$CHECKPOINT_FILE" "$CHECKPOINT_URL"
    echo "Checkpoint salvato con successo in $CHECKPOINT_FILE"
else
    echo "[4/5] Checkpoint base $CHECKPOINT_FILE già presente."
fi

# Determinazione dell'eseguibile Python (preferenza uv / .venv rispetto al python di sistema)
if command -v uv &> /dev/null; then
    PYTHON_EXEC="uv run --no-sync python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# 6. Verifica installazione
echo "[5/5] Verifica dell'ambiente JAX e caricamento checkpoint..."
$PYTHON_EXEC -c "
import jax
import pickle
print(f'JAX Version: {jax.__version__}')
print(f'Dispositivi disponibili: {jax.devices()}')
try:
    with open('$CHECKPOINT_FILE', 'rb') as f:
        data = pickle.load(f)
    print('Checkpoint Ithaca v1 caricato con successo!')
    if 'params' in data:
        print(f'Parametri top-level: {list(data[\"params\"].keys())}')
except Exception as e:
    print(f'Nota durante la lettura del pickle: {e}')
"

echo "Setup completato! Pronto per la preparazione dati e fine-tuning."

